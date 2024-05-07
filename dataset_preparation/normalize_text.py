import re

from utils import *


def main():
    inp_meta_fp = '../metas/v2/metadata_splitted_unnormalized.json'
    new_meta_fp = inp_meta_fp.replace('_unnormalized.json', '.json')

    normalize_utterance_annotation(inp_meta_fp, new_meta_fp)


def normalize_utterance_annotation(inp_meta_fp, new_meta_fp):
    '''
    Normalize the text annotation inside a metadata json file
    Save the normalized file to new_meta_fp
    '''
    metadata = read_json(inp_meta_fp)
    for utterance in metadata:
        lyric = metadata[utterance]['lyrics']
        normalized_lyric = normalize_an_utterance(lyric)
        metadata[utterance]['lyrics'] = normalized_lyric

    clean_meta = {}
    duration_threshold = timedelta(seconds=28)
    for id in metadata:
        if timecode_to_timedelta(metadata[id]['duration']) > duration_threshold:
            print('too long')
            continue
        if len(metadata[id]['lyrics']) == 0:
            print('no valid lyric')
            continue
        clean_meta[id] = metadata[id]
    
    save_json(clean_meta, new_meta_fp)


def normalize_an_utterance(utterance):
    # 1. Remove blank
    utterance = utterance.strip()

    # 2. uppercase
    utterance = utterance.upper()

    # 3. numbers to text: no need

    # 4. remove ',', '.', '"', ':', '-'
    utterance = utterance.replace(',', ' ')
    utterance = utterance.replace('.', ' ')
    utterance = utterance.replace('"', ' ')
    utterance = utterance.replace(':', ' ')
    utterance = utterance.replace('-', '')

    # 5. (oh-oh-oh) -> oh oh oh
    utterance = utterance.replace('(OH-OH-OH)', 'OH OH OH')

    # 6. do-re-mi -> do re mi
    utterance = utterance.replace('DO-RE-MI', 'DO RE MI')

    # 7. A-be-see -> a b c
    utterance = utterance.replace('A-BE-SEE', 'A B C')

    # 8. Remove original text in deletion: (deleted words)
    if '(' in utterance or ')' in utterance:
        # utterance = remove_parentheses(utterance)
        utterance = utterance.replace('(', '')
        utterance = utterance.replace(')', '')

    # 9. Remove insertion mark: '{' and '}'
    utterance = utterance.replace('{', '')
    utterance = utterance.replace('}', '')

    # 10. Remove substitution mark and original words: [original words]
    if '[' in utterance or ']' in utterance:
        # utterance = remove_square_brackets(utterance)
        utterance = utterance.replace('[', '')
        utterance = utterance.replace(']', '')

    # 11. Remove bad pronunciation mark: '/'
    utterance = utterance.replace('/', '')

    # 12. remove punctuation
    utterance = utterance.replace('-', ' ')
    utterance = re.sub("[^A-Z' ]", "", utterance)

    # 13. IN' -> ING
    utterance = utterance.replace('IN\'', 'ING')

    # ev'ry -> every
    utterance = utterance.replace("ev'ry", "every")

    # Finally, remove space
    t = utterance.strip().split(' ')
    utterance = ' '.join([i for i in t if len(i) > 0])

    return utterance


def remove_parentheses(line):
    '''
    Remove the bracket and the content inside the bracket from a sentence
    '''
    left_bracket_index = line.index('(')
    right_bracket_index = line.index(')')
    inside_bracket = line[left_bracket_index:right_bracket_index + 1]
    t = [i.strip() for i in line.split(inside_bracket)]
    ret = ' '.join(t)
    return ret


def remove_square_brackets(line):
    '''
    Remove the bracket and the content inside the bracket from a sentence
    '''
    left_bracket_index = line.index('[')
    right_bracket_index = line.index(']')
    inside_bracket = line[left_bracket_index:right_bracket_index + 1]
    t = [i.strip() for i in line.split(inside_bracket)]
    ret = ' '.join(t)
    return ret


def check_numbers():
    '''
    See if there exists numbers in text
    Result: no
    '''
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # dataset_root = '../../dataset_v1.0/utterance_level/'
    meta_path = '../metadata.json'
    metadata = read_json(meta_path)
    for utterance in metadata:
        print(utterance)
        lyric = metadata[utterance]['lyrics']
        t = lyric.strip().split(' ')
        for i in t:
            if i in numbers:
                raise Exception('Number found in {}'.format(utterance))


def check_hyphen():
    '''
    See if there exists numbers in text
    Result: no
    '''
    # dataset_root = '../../dataset_v1.0/utterance_level/'
    meta_path = '../metadata.json'
    metadata = read_json(meta_path)
    for utterance in metadata:
        # print(utterance)
        lyric = metadata[utterance]['lyrics']
        if ' - ' in lyric:
            raise Exception('Number found in {}: {}'.format(utterance, lyric))


if __name__ == '__main__':
    main()
