from utils import *

def main():
    statistics()

def statistics():
        '''
        统计数据集的utterance number, total duration, avg utt duration
        '''
        meta = read_json('../metadata.json')
        num_utter = len(meta)
        total_duration = timedelta(seconds=0)
        for id in meta:
            duration = timecode_to_timedelta(meta[id]['duration'])
            total_duration += duration
        print(num_utter, total_duration, total_duration / num_utter)

if __name__ == '__main__':
    main()