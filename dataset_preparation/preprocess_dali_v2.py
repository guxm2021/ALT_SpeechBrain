from utils import *
from download_dali_v2 import download_audio_for_dali_v2
from demucs.separate import *
from normalize_text import normalize_utterance_annotation

def main():
    procedure_v2()

# Change the variable to path to official dali annotation path
dali_data_path = '/data1/guxm/asr_datasets/DALI/v2/annotations/annot_tismir' # only used by function extract_utter_annotation_all_audio
necessary_meta_fp = './misc/dali_v2_utter_annotation_full.json'
clean_meta_fp = './raw/dali/metas/utter_annotation_downloaded_english.json'
dataset_download_dir = './downloads/dali_v2/'
audio_dir = jpath(dataset_download_dir, 'audio')
audio_ss_dir = './raw/dali/audio_ss'
segment_out_dir = './segmented/dali_v2'
utter_meta_fp = jpath(segment_out_dir, 'metadata_unnormalized.json')
final_meta_fp = jpath(segment_out_dir, 'metadata.json')

def procedure_v2():
    '''
    1. Obtain useful info from raw dataset, create song-level metadata
    2. Download the audio
    3. Delete songs that are not in English, and not downloaded (from where? TODO)
    4. Delete illegal utterances in annotation (start time >= end time)
    5. Segment the dataset to utterance level
    6. Generate metadata.json for utterance-level dataset
    7. Split dataset
    8. Normalize the textual annotation
    '''

    # extract_utter_annotation_all_audio(dali_data_path)  # We have make the output ready, at './misc/dali_v2_utter_annotation_full.json'
    download_audio_for_dali_v2(meta_fp=necessary_meta_fp)     # Download audios for the dataset
    source_separate_for_dali()                          # Get the vocal part
    update_utter_annotation()                           # Delete un-downloaded entries from metadata
    validate_annotation()                               # Ensure all utterance annotations are legal
    segment_by_annotation()                             # Segment to utterance-level dataset
    deduplicate()                                       # Remove any test set content from DALI v2 for training
    split_dataset()                                     # Prepare a development set (validation set)
    normalize_annotation()

def normalize_annotation():
    '''
    Do text normalization to the metadata
    '''
    normalize_utterance_annotation(inp_meta_fp=utter_meta_fp, new_meta_fp=final_meta_fp)


def deduplicate():
    '''
    Some songs in the test dataset, the DALI-Test (a subset of DALI v1), also shows up in DALI v2.
    Remove any songs in the test set from DALI v2 (which is used for training).

    Please implement this function yourself according to your test set.

    Note: 
        Ids of a same audio can be different between DALI v1 and v2. Do not use it as measure.  
        Use lyric content to determine data leakage instead.
    '''
    raise NotImplementedError

def source_separate_for_dali():
    '''
    This function read the configuration for Demucs, and 
    Conduct source separation for the dali dataset using demucs
    '''
    # Create dir for output
    create_dirs_if_not_exist(audio_ss_dir)

    # Call the demucs to do the separation
    args = read_json('./misc/default_args.json')
    args = argparse.Namespace(**args)
    demucs_ss(audio_dir, audio_ss_dir, args)


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
    for track_name in pbar:
        # Skip .DS_Store
        if track_name.startswith('.'):
            continue

        pbar.set_description('Processing {}'.format(track_name))
        track_dir = jpath(audio_dir_root, track_name)

        # Skip empty folders
        files = ls(track_dir)
        n_files = len(files)
        if n_files == 0:
            continue

        assert n_files == 1
        audio_fn = files[0]
        audio_name = audio_fn.strip().split('.')[0]
        inp_audio_fp = jpath(audio_dir_root, track_name, audio_fn)
        output_folder = jpath(dali_ss_root, track_name)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if len(ls(output_folder)) == 1:  # if already have output, skip this file
            print('Skip, output already generated')
            continue 
        output_path = jpath(output_folder, audio_name)

        # Separate vocal part from the wav file
        try:
            print(f"Separating track {inp_audio_fp}")
            if not os.path.isfile(inp_audio_fp):
                raise Exception(f'Input audio path {inp_audio_fp} not valid')

            wav = load_track(inp_audio_fp, model.audio_channels, model.samplerate)

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
            stem = jpath(output_folder, 'vocal.wav')  # output path
            save_audio(sources.pop(
                model.sources.index(args.stem)), stem, **kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            exit(10)

def split_dataset():
    '''
    Add "split" label to each entry of metadata_non_eval.json
    Save as "metadata.json"
    '''
    # The file below was a random subset from all track ids inside dali v2
    # Generated by code dev_ids = sample(ids, 24)
    dev_ids_fp = './misc/dali_v2_dev_ids.json'
    dev_ids = read_json(dev_ids_fp)
    meta = read_json(utter_meta_fp)

    for id in meta:
        entry = meta[id]
        audio_id = id.split('-')[0]
        if audio_id in dev_ids:
            entry['split'] = 'valid'
        else:
            entry['split'] = 'train'
    
    save_json(meta, utter_meta_fp)
    

def segment_by_annotation():
    '''
    Segment all audios by annotation. Meanwhile, construct metadata.json
    
    '''
    target_folder = jpath(segment_out_dir, 'data_seg') # save segmentation of original wav file
    resample_folder = jpath(segment_out_dir, 'data')  # save 16 kHz wav
    create_dirs_if_not_exist(segment_out_dir)
    create_dirs_if_not_exist(target_folder)
    create_dirs_if_not_exist(resample_folder)

    utter_annotation = read_json(clean_meta_fp)

    meta = {}
    pbar = tqdm(utter_annotation)
    for audio_name in pbar:
        pbar.set_description('Processing {}'.format(audio_name))
        segmentor = Segmenter(audio_name=audio_name, 
                              audio_folder=audio_ss_dir, 
                              target_folder=target_folder,
                              resample_folder=resample_folder,
                              annotation=utter_annotation[audio_name]['annotation'])
        meta = segmentor.segment(meta)
    save_json(meta, utter_meta_fp)


class Segmenter:
    '''
    Each Segmenter obj is only responsible for segment one audio file
    '''

    def __init__(self, audio_name, audio_folder, target_folder, resample_folder, annotation):
        '''
        Construct segmenter
        :param audio_folder: folder containing whole song wav files
        :param target_folder: folder to contain segmentated audios
        :param annotation_folder:
        '''
        self.annotation = annotation
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
            entry = self.annotation[i]
            start_time = entry['start']
            end_time = entry['end']
            lyrics = entry['lyric']
            start_time_code = start_time
            end_time_code = end_time
            utterance_name = self.audio_name + '-{:02d}.wav'.format(i)
            ''' Example command: ffmpeg -i mic.wav -ss 00:00:05.010 -to 00:00:20 -ar 16000 output.wav '''
            cmd = 'ffmpeg -y -i {} -ss {} -to {} -ar 16000 {} -loglevel error'.format(
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
            # resampled_waveform = torchaudio.functional.resample(waveform, sample_rate, resample_rate)
            # resampled_waveform = torchaudio.compliance.kaldi.resample_waveform(waveform, sample_rate, resample_rate)
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

def validate_annotation():
    '''
    分割的时候发现有些utterance的end time竟然小于start time, 检查一下
    Filter out such utterances from annotation
    '''
    utter_annotation = read_json(clean_meta_fp)

    for id in utter_annotation:
        annotation = utter_annotation[id]['annotation']
        annotation_clean = []
        for entry in annotation:
            if timecode_to_timedelta(entry['start']) < timecode_to_timedelta(entry['end']):
                annotation_clean.append(entry)
            else:
                print(entry['start'], entry['end'])
        utter_annotation[id]['annotation'] = annotation_clean
    save_json(utter_annotation, clean_meta_fp)

def get_undownloaded_ids():
    '''
    保存未能下载的audio ids
    '''
    utter_annotation = read_json('../metas/v2/utter_annotation_full.json')
    audio_path = '/data1/guxm/asr_datasets/DALI/v2/audio_ss'
    # downloaded_ids = set(ls(audio_path))
    ids = []
    for id in utter_annotation:
        t = len(ls(jpath(audio_path, id)))
        if t == 0:
            ids.append(id)
        elif t == 1:
            pass
        else:
            raise Exception('incorrect tgt folder')
  
    print(len(ids))
    save_json(ids, '../metas/v2/unavailable_ids_dali_v2.json')

def update_utter_annotation():
    '''
    Generate a new metadata file ""
    从所有audio的utter_annotation中去掉
    1. 未下载的歌曲
    2. 不是英语的歌曲 
    '''
    meta_root = './raw/dali/metas'
    full_meta_fp = jpath(meta_root, 'utter_annotation_full.json')
    full_meta = read_json(full_meta_fp)

    audio_dir = './raw/dali/audio_ss'

    downloaded_ids = set(ls(audio_dir))
    ret = {}
    for id in full_meta:
        if id in downloaded_ids and full_meta[id]['language'] == 'english':
            ret[id] = full_meta[id]
    
    save_fp = clean_meta_fp
    save_json(ret, save_fp)


def extract_utter_annotation_all_audio(dali_path):
    '''
    Extract useful info from raw DALI metadata
    '''
    import DALI as dali_code

    save_meta_fp = '../metas/v2/utter_annotation_full.json'

    dali_data = dali_code.get_the_DALI_dataset(dali_path, skip=[], keep=[])
    res = {}
    for id in dali_data:
        _, utter_annot = extract_utter_annotation_one_audio(dali_data[id])
        res[id] = utter_annot
    save_json(res, '../metas/v2/utter_annotation_full.json')


def extract_utter_annotation_one_audio(input):
    '''
    input: DALI metadata entry
    return: metadata entry of myself

        'id': id
        {
            'language': language
            'title': song title (lower case)
            'artist': song artist (lower case)
            'url': youtube url
            'annotation': [
                {
                    'start': start timecode
                    'end': end timecode
                    'lyric': utterance lyrics
                },
                ...
            ]
        }

    '''
    id = input.info['id']
    language = input.info['metadata']['language']
    utter_annotation = []
    raw_annotation = input.annotations['annot']['lines']
    title = input.info['title'].lower()
    artist = input.info['artist'].lower()
    url = input.info['audio']['url']
    for line in raw_annotation:
        lyric = line['text']
        start = timedelta(seconds=float(line['time'][0]))
        end = timedelta(seconds=float(line['time'][1]))
        start_timecode = timedelta_to_timecode(start)
        end_timecode = timedelta_to_timecode(end)
        utter_annotation.append({
            'start': start_timecode,
            'end': end_timecode,
            'lyric': lyric
        })
    return id, {'language': language, 'title': title, 'artist': artist, 'url': url, 'annotation': utter_annotation}


if __name__ == '__main__':
    main()
