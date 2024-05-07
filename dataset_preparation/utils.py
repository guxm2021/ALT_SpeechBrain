import os
import json
import torchaudio
import torchaudio.transforms as T
from datetime import timedelta
from tqdm import tqdm

jpath = os.path.join
ls = os.listdir

def main():
    pass

def read_json(path):
    with open(path, 'r', encoding='utf8') as f:
        data = f.read()
    data = json.loads(data)
    return data


def save_json(data, path):
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=False))


def print_json(data):
    '''
    Format print a json string
    '''
    print(json.dumps(data, indent=4, ensure_ascii=False))

def timecode_to_timedelta(timecode):
    '''
    Convert timecode 'MM:SS.XXX' to timedelta object
    '''
    m, s = timecode.strip().split(':')
    m = int(m)
    s = float(s)
    ret = timedelta(minutes=m, seconds=s)
    return ret

def timedelta_to_timecode(delta):
    time_str = str(delta)
    t = time_str.split('.')[-1]
    if len(t) == 6:
        time_code = time_str[:-3]
    else:
        time_code = '{}.000'.format(time_str)

    t = time_code.split(':')
    if len(t) == 3:
        time_code = ':'.join(t[1:])

    return time_code

def timecode_to_millisecond(timecode):
    m, s = timecode.split(':')
    s, ms = s.split('.')
    m = int(m)
    s = int(s)
    ms = int(ms)
    s += m * 60
    ms += s * 1000
    return ms

def _get_sample(path, resample=None):
    effects = [
        ["remix", "1"]
    ]
    if resample:
        effects.extend([
            ["lowpass", f"{resample // 2}"],
            ["rate", f'{resample}'],
        ])
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_sample(path, resample=None):
    return _get_sample(path, resample=resample)

def create_dirs_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
if __name__ == '__main__':
    main()