
"""
Data preparation for Hansen / Jamendo / Mauch Datasets

Authors
* Xiangming Gu 2022
"""
import re
import os
import csv
import random
import json
import argparse
from collections import Counter
import logging
import torchaudio
from tqdm import tqdm
logger = logging.getLogger(__name__)
SAMPLERATE = 16000
DATASETS = ["Hansen", "Jamendo", "Mauch"]


def prepare_audio_eval(
    root,
    save_folder,
    dataset_name,
    skip_prep = False,
):
    """
    This function prepares the csv files for Hansen / Jamendo / Mauch Datasets
    """
    if skip_prep:
        return
    assert dataset_name in DATASETS
    data_folder = os.path.join(root, 'data')
    anno_path = os.path.join(root, 'metadata.json')
    save_csv = os.path.join(save_folder, dataset_name + '.csv')
    print(f"Save data in the path: {save_csv}")
    
    # open json files
    with open(anno_path, 'r') as f:
        data = json.load(f)
    f.close()

    csv_lines = [["ID", "duration", "wav", "wrd"]]

    for key in tqdm(data.keys()):
        # fetch values
        value = data[key]
        wrds = value["lyrics"]

        # determine target path
        path = os.path.join(data_folder, key + '.wav')

        # load audio
        signal, fs = torchaudio.load(path)
        assert fs == SAMPLERATE
        duration = signal.shape[1] / SAMPLERATE
        if duration < 0.1:
            continue
        
        # construct csv files
        csv_line = [
            key, str(duration), path, wrds,
        ]

        # append
        csv_lines.append(csv_line)

    # create csv files for each split
    
    csv_save_train = os.path.join(save_csv)
    with open(csv_save_train, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_lines:
            csv_writer.writerow(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hansen_folder", type=str, default="/path/to/hansen", help="The saved path for Hansen dataset")
    parser.add_argument("--jamendo_folder", type=str, default="/path/to/jamendo", help="The saved path for Jamendo dataset")
    parser.add_argument("--mauch_folder", type=str, default="/path/to/jamendo", help="The saved path for Mauch dataset")
    parser.add_argument("--save_folder", type=str, default="data", help="The saved path for prepared csv files")
    args = parser.parse_args()
    prepare_audio_eval(root=args.hansen_folder, save_folder=args.save_folder, dataset_name="Hansen")
    prepare_audio_eval(root=args.jamendo_folder, save_folder=args.save_folder, dataset_name="Jamendo")
    prepare_audio_eval(root=args.mauch_folder, save_folder=args.save_folder, dataset_name="Mauch")