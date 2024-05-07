from utils import *
from demucs.separate import *
from normalize_text import normalize_utterance_annotation

def main():
    procedures()

dataset_dir = './raw/jamendo'
utter_annot_dir = jpath(dataset_dir, 'annotations_utter')
ori_audio_dir = jpath(dataset_dir, 'audio')
ss_audio_dir = jpath(dataset_dir, 'audio_ss')
vocal_end_time_fp = './misc/jamendo_vocal_end_time.json'
ori_annot_dir = jpath(dataset_dir, 'annotations')
ori_lyric_dir = jpath(dataset_dir, 'lyrics')
segmented_dataset_root = './segmented/jamendo'
metadata_fp = jpath(segmented_dataset_root, 'metadata.json')
utter_audio_dir = jpath(segmented_dataset_root, 'data_seg') # save segmentation of original wav file
utter_audio_resample_dir = jpath(segmented_dataset_root, 'data')  # save 16 kHz wav
normalized_meta_fp = jpath(segmented_dataset_root, 'metadata.json')

def procedures():
    source_separate_for_jamendo()
    get_utter_annotation()
    segment_by_annotation()
    normalize_annotation()

def normalize_annotation():
    '''
    Do text normalization to the metadata
    '''
    normalize_utterance_annotation(inp_meta_fp=metadata_fp, new_meta_fp=normalized_meta_fp)


def source_separate_for_jamendo():
    '''
    This function read the configuration for Demucs, and 
    Conduct source separation for the mauch dataset using demucs
    '''
    create_dirs_if_not_exist(ss_audio_dir)

    args = read_json('./misc/default_args.json')
    args = argparse.Namespace(**args)
    demucs_ss(ori_audio_dir, ss_audio_dir, args)


def demucs_ss(src_root, tgt_root, args):
    '''
    This function do the 2-stem separation for audio datasets,
    with the Demucs model
    '''
    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if args.segment is not None and args.segment < 8:
        fatal('Segment must greater than 8. ')

    if isinstance(model, BagOfModels):
        print(f"Selected model is a bag of {len(model.models)} models. "
              "You will see that many progress bars per track.")
        if args.segment is not None:
            for sub in model.models:
                sub.segment = args.segment
    else:
        if args.segment is not None:
            sub.segment = args.segment

    model.cpu()
    model.eval()

    if args.stem is not None and args.stem not in model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                stem=args.stem, sources=', '.join(model.sources)))
    out_dir = jpath(args.out, args.name)
    create_dirs_if_not_exist(out_dir)
    # out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out_dir}")

    # Create input path list for DALI dataset
    print(args)
    input_list = []
    audio_dir_root = src_root
    dali_ss_root = tgt_root
    songs = ls(audio_dir_root)
    # cnt = 0
    from tqdm import tqdm
    pbar = tqdm(songs)
    abnormal_files = {}
    for mp3 in pbar:
        pbar.set_description('Processing {}'.format(mp3))
        audio_path = jpath(audio_dir_root, mp3)
        id = mp3[:-4]
        output_folder = jpath(dali_ss_root, id)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if len(ls(output_folder)) == 1:  # if already have output, skip this file
            print('Skip, output already generated')
            continue 
        song_path = audio_path
        output_path = jpath(output_folder, id)

        # Separate vocal part from the wav file
        try:
            track = song_path  # input file path
            print(f"Separating track {track}")
            wav = load_track(track, model.audio_channels, model.samplerate)

            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            sources = apply_model(model, wav[None], device=args.device, shifts=args.shifts,
                                  split=args.split, overlap=args.overlap, progress=True,
                                  num_workers=args.jobs)[0]
            sources = sources * ref.std() + ref.mean()

            # track_folder = out / track.name.rsplit(".", 1)[0]
            # track_folder.mkdir(exist_ok=True)
            if args.mp3:
                ext = ".mp3"
            else:
                ext = ".wav"
            kwargs = {
                'samplerate': model.samplerate,
                'bitrate': args.mp3_bitrate,
                'clip': args.clip_mode,
                'as_float': args.float32,
                'bits_per_sample': 24 if args.int24 else 16,
            }

            sources = list(sources)
            print(len(sources))
            # stem = str(track_folder / (args.stem + ext))
            stem = jpath(output_folder, 'vocal.wav')  # output path
            # save_audio(sources.pop(model.sources.index(args.stem)), stem, **kwargs)
            save_audio(sources.pop(
                model.sources.index(args.stem)), stem, **kwargs)
        except Exception as e:
            print(e)

    # save_json(abnormal_files, 'abnormal_files.pkl')
    # torch.save(abnormal_files, 'abnormal_files_test.pkl')


class Segmenter:
    '''
    Each Segmenter obj is only responsible for segment one audio file
    '''

    def __init__(self, audio_name, audio_folder, target_folder, resample_folder, annotation_path):
        '''
        Construct segmenter
        :param audio_folder: folder containing whole song wav files
        :param target_folder: folder to contain segmentated audios
        :param annotation_folder:
        '''
        self.annotation = read_json(annotation_path)
        self.utterance_num = len(self.annotation)
        self.target_folder = target_folder
        self.audio_path = jpath(audio_folder, '{}/vocal.wav'.format(audio_name))
        self.audio_name = audio_name
        self.resample_folder = resample_folder

    def segment(self, meta):
        '''
        Perform segmentation
        '''
        self.create_segmentation_folders()
        meta = self.segment_audio(meta)
        return meta

    def create_segmentation_folders(self):
        '''
        Create 2-digit folders for utterance segmentations
        '''
        segment_folder = self.target_folder
        if not os.path.exists(segment_folder):
            os.mkdir(segment_folder)
        if not os.path.exists(self.resample_folder):
            os.mkdir(self.resample_folder)

    def segment_audio(self, meta):
        '''
        Get all utterance segments of one audio recording, and save them to corresponding utterance folders
        Meanwhile, add metadata of this song to argument meta
        '''
        for i in range(self.utterance_num):
            start_time, end_time, lyrics = self.annotation[i]
            start_time_code = start_time
            end_time_code = end_time
            utterance_name = self.audio_name + '-{:02d}.wav'.format(i)
            ''' Example command: ffmpeg -i mic.wav -ss 00:00:05.010 -to 00:00:20 -c copy output.wav '''
            cmd = 'ffmpeg -y -i {} -ss {} -to {} -c copy {} -loglevel error'.format(
                self.audio_path, start_time_code, end_time_code,
                os.path.join(self.target_folder, utterance_name)
            )
            os.system(cmd)

            # Downsample to 16 kHz
            sample_rate = 44100
            resample_rate = 16000
            waveform, sr = torchaudio.load(jpath(self.target_folder, utterance_name))
            resampler = T.Resample(sample_rate, resample_rate)
            resampled_waveform = resampler(waveform)
            torchaudio.save(jpath(self.resample_folder, utterance_name), resampled_waveform, resample_rate)

            # Add entry to meta
            # print(end_time, start_time)
            utter_id = utterance_name[:-4]
            meta[utter_id] = {
                'path': jpath('data', utterance_name).replace('\\', '/'),
                'duration': timedelta_to_timecode(
                    timecode_to_timedelta(end_time) - timecode_to_timedelta(start_time)),
                'lyrics': lyrics
            }
        return meta


def get_utter_annotation():
    '''
    Convert the original word-level annotations to utterance-level annotation.

    Obtain word-level annotation from {path_to_dataset}/wordonsets/ and {path_to_dataset}/lyrics/
    Save utterance annotation to {path_to_dataset}/annotations_utter/
    '''
    create_dirs_if_not_exist(utter_annot_dir)

    # Read the vocal end time file content
    vocal_end_time = read_json(vocal_end_time_fp)

    lyric_files = ls(ori_lyric_dir)
    annotations = ls(ori_annot_dir)
    audio_names = [t.split('.wordonset')[0] for t in annotations]
    pbar = tqdm(audio_names)
    for audio_name in pbar:
        pbar.set_description(audio_name)
        
        lyric_path = jpath(ori_lyric_dir, '{}.raw.txt'.format(audio_name))
        annotation_path = jpath(ori_annot_dir, '{}.wordonset.txt'.format(audio_name))
        lyric = read_lyric_file(lyric_path)
        annotation = read_annotation_file(annotation_path)

        assert len(lyric) == len(annotation)
        for i in range(len(lyric)):
            sec = annotation[i]
            lyric[i].append(timedelta(seconds=sec))

        # Get offset time
        offset_timecode = vocal_end_time[audio_name]
        offset_timedelta = timecode_to_timedelta(offset_timecode)

        # Convert word-level annotation to utterance-level
        utter_annotation = word2utter(lyric, offset_timedelta)

        # Save
        save_json(utter_annotation, jpath(utter_annot_dir, '{}.json'.format(audio_name)))



def word2utter(lyric, offset):
    '''
    Convert word-level annotation to utterance-level annotation, for annotation of one audio
    :param lyric [ [word, line_number, onset_time], ... ]
    :param offset timedelta, offset time of last word
    :return [[{start_time_code}, {end_time_code}, {utter_lyrics}], ... ]
    '''
    ret = []
    line_num = 0
    utter_entry = None
    last_onset = None  # onset time of last word in one utterance
    for i, word_entry in enumerate(lyric):
        word, word_line_num, word_onset = word_entry
        if line_num != word_line_num:
            if utter_entry != None:
                assert word_onset > last_onset
                last_word_duration_threshold = 2
                if (word_onset - last_onset) > timedelta(seconds=last_word_duration_threshold):
                    utter_offset = last_onset + timedelta(seconds=last_word_duration_threshold)
                else:
                    utter_offset = word_onset
                utter_entry[1] = utter_offset
                ret.append(utter_entry)
            utter_entry = [None, None, None]
            utter_entry[0] = word_onset
            last_onset = word_onset
            utter_entry[2] = word
            line_num = word_line_num
        else:
            last_onset = word_onset
            utter_entry[2] += ' {}'.format(word)
    if utter_entry != None:
        utter_entry[1] = offset
        ret.append(utter_entry)
    for entry in ret:
        entry[0] = timedelta_to_timecode(entry[0])
        entry[1] = timedelta_to_timecode(entry[1])
    return ret


def read_offset_file(path):
    '''
    有一个文件记录了每首歌最后一个单词的offset time
    :return str, offset time (timecode)
    '''
    with open(path, 'r') as f:
        text = f.read()
    return text


def read_annotation_file(path):
    '''
    Read annotation to a list
    '''
    with open(path, 'r', encoding='utf8') as f:
        text = f.readlines()
    text = [line.strip() for line in text]
    text = [line for line in text if len(line) > 0]
    ret = []
    for line in text:
        onset_time = float(line.strip())
        ret.append(onset_time)
    return ret


def read_lyric_file(path):
    '''
    Read lyric file, split each word, and assign the line number to each word
    :return: [(word, line_num)]
    '''
    try:
        with open(path, 'r', encoding='utf8') as f:
            text = f.readlines()
    except Exception as e:
        print(e)
        print(path)
        exit(100)
    text = [line.strip() for line in text]
    text = [line for line in text if len(line) > 0]
    ret = []
    for i, line in enumerate(text):
        t = line.split(' ')
        for word in t:
            word = word.strip()
            if word == '':
                continue
            ret.append([word, i + 1])
    return ret

def segment_by_annotation():
    '''
    Segment all audios from song-level to utterance-level, according to the utterance-level annotation, 
    meanwhile, construct metadata.json

    metadata.json looks like:
    {
    "{song_name}-{utter_id}": {
        "path": "data/{song_name}-{utter_id}",
        "duration": "0:02.422",
        "lyrics": "GREATNESS AS YOU"
        }
        ...
    }
    '''
    create_dirs_if_not_exist(segmented_dataset_root)

    annotation_files = ls(utter_annot_dir)
    song_names = [i[:-5] for i in annotation_files]

    meta = {}
    pbar = tqdm(song_names)
    for audio_name in pbar:
        pbar.set_description('Processing {}'.format(audio_name))
        segmentor = Segmenter(
            audio_name=audio_name, 
            audio_folder=ss_audio_dir, 
            target_folder=utter_audio_dir,
            resample_folder=utter_audio_resample_dir,
            annotation_path='{}/{}.json'.format(utter_annot_dir, audio_name)
        )
        meta = segmentor.segment(meta)
    save_json(meta, metadata_fp)


if __name__ == '__main__':
    main()
