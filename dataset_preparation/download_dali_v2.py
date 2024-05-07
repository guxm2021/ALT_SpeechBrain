
from utils import *
import os
from tqdm import tqdm

import yt_dlp

def main():
    download_audio_for_dali_v2()

def download_audio_for_dali_v2(meta_fp):
    meta_fp = meta_fp
    save_audio_folder = './downloads/dali_v2/audio'
    error_fp = './download_v2_error.json'

    os.makedirs(save_audio_folder, exist_ok=True)
    meta = read_json(meta_fp)
    errors = get_audio_v2(meta, save_audio_folder, skip=[], keep=[])
    save_json(errors, error_fp)

def get_audio_v2(meta, path_output, skip=[], keep=[]):
    """Get the audio for the dali dataset.

    It can download the whole dataset or only a subset of the dataset
    by providing either the ids to skip or the ids that to load.

    Parameters
    ----------
        dali_info : list
            where elements are ['DALI_ID', 'NAME', 'YOUTUBE', 'WORKING']
        path_output : str
            full path for storing the audio
        skip : list
            list with the ids to be skipped.
        keep : list
            list with the ids to be keeped.
    """
    errors = []
    print(f"path_output: {path_output}")
    for i in tqdm(meta):
        entry = meta[i]
        audio_from_url(url=entry['url'], name=i, path_output=path_output, errors=errors)
    return errors

def audio_from_url(url, name, path_output, errors=[]):
    """
    Download audio from a url.
        url : str
            url of the video (after watch?v= in youtube)
        name : str
            used to store the data
        path_output : str
            path for storing the data
    """
    error = None

    # ydl(youtube_dl.YoutubeDL): extractor
    ydl = get_my_ydl_new(name, path_output)

    # ydl.params['outtmpl'] = ydl.params['outtmpl'] % {
    #     'ext': ydl.params['postprocessors'][0]['preferredcodec'],
    #     'title': name}

    base_url = 'http://www.youtube.com/watch?v='
    if ydl:
        print ("Downloading " + url)
        try:
            ydl.download([base_url + url])
            # ydl.extract_info([base_url + url])
        except Exception as e:
            print(e)
            error = e
    if error:
        errors.append([name, url, error])
    return

class MyLogger(object):
    def debug(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

def get_my_ydl_new(name, directory=os.path.dirname(os.path.abspath(__file__))):
    ydl = None
    outtmpl = None
    if os.path.exists(directory):
        os.makedirs(os.path.join(directory, name), exist_ok=True)
        outtmpl = os.path.join(directory, name, '%(title)s.%(ext)s')

        ydl_opts = {'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio',
                                        'preferredcodec': 'mp3',
                                        'preferredquality': '320'}],
                    'outtmpl': outtmpl,
                    'logger': MyLogger(),
                    'progress_hooks': [my_hook],
                    'verbose': False,
                    'ignoreerrors': False,
                    'external_downloader': 'ffmpeg',
                    'nocheckcertificate': True}
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        ydl.cache.remove()
        import time
        time.sleep(.5)

    return ydl



if __name__ == '__main__':
    main()