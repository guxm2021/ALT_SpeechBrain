"""
Data preparation for DALI datasets

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


def prepare_text_dali(
    root,
    save_folder,
    skip_prep = False,
):
    """
    This function prepares the text corpora valid and train splits of DALI
    """
    if skip_prep:
        return
    anno_path = os.path.join(root, 'metadata.json')
    
    # save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # open json files
    with open(anno_path, 'r') as f:
        data = json.load(f)
    f.close()

    train_text = []
    dev_text = []

    for key in tqdm(data.keys()):
        # fetch values
        value = data[key]
        # source_path = value["path"]   # data/<split>/<name>
        split = value["split"]
        wrds = value["lyrics"]

        # remove extra blank spaces
        wrds = wrds.split(' ')
        wrds = list(filter(None, wrds))
        if len(wrds) == 0:
            print("No targets")
            continue
        wrds = ' '.join(wrds)

        # append
        if split == "train":
            train_text.append(wrds)
        elif split == "valid":
            dev_text.append(wrds)

    # create csv files for each split
    txt_save_train = os.path.join(save_folder, "train.txt")
    with open(txt_save_train, mode="w") as f:
        for line in train_text:
            f.write(line)
            f.write('\n')
    
    txt_save_dev = os.path.join(save_folder, "valid.txt")
    with open(txt_save_dev, mode="w") as f:
        for line in dev_text:
            f.write(line)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/path/to/dali", help="The saved path for DALI data folder")
    parser.add_argument("--save_folder", type=str, default="data/dali", help="The saved path for prepared text corpora")
    args = parser.parse_args()
    prepare_text_dali(root=args.data_folder, save_folder=args.save_folder)
    