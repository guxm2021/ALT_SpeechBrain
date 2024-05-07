# Dataset Preparation

This directory contains code to prepare datasets for training and evaluation of the model.

## Obtaining Dataset

**IMPORTANT NOTICE**: In this section, **only the bolded links are officially public accessible**. For download links in *italics*, you may only click on the provided links after you have read and agreed to the [confidentiality agreement](https://docs.google.com/document/d/1cgKaLV7_edwlKk04zephPaXmHuVRbUQpQoCnUsVidSo/edit?usp=sharing).

- DSing: please follow its [**github repo**](https://github.com/groadabike/Kaldi-Dsing-task)
- DALI: 
    - DALI v2: [**[annotation](https://zenodo.org/records/3576083)** (need request)]
    - DALI Test: Follow the recipe in this [**github repo**](https://github.com/emirdemirel/DALI-TestSet4ALT)
- Hansen: [[**author: Jens Kofod Hansen**](mailto:jens@kofod-hansen.com)] [[*audio*](https://www.dropbox.com/s/asfi22fng4cekex/fullsongs.zip?dl=0)] [[*annotation*](https://www.dropbox.com/s/nxiigp3qgfqeq0r/phonemeannotations.zip?dl=0)] 
- Jamendo: [[**audio and annotation**](https://github.com/f90/jamendolyrics)]
- Mauch: [[**author: Matthias Mauch**](mailto:mail@matthiasmauch.net)] [[*unofficial*](https://drive.google.com/file/d/1KyEDnDbdz6vXjCNINXIqUej2GJgmr0L7/view?usp=sharing)]

Note:
- Obtaining DALI v2 official annotation is optional. We have get the necessary info from metadata ready in the code.
- The unofficial version of Mauch dataset also contains a copy of unofficial Hansen dataset.

## Preprocess the data

### DSing
For DSing dataset, we followed the official procedure to preprocess it.

### DALI
For remaining dataset, including DALI, Hansen, Jamendo, Mauch, we preprocess each of them to generate (1) a json-format metadata file, containing utterance-level annotation, and (2) a folder containing utterance-level audio files.

We have get the lyric annotation from original dataset ready, at `./misc/dali_v2_utter_annotation_full.json`. So if you don't need additional info from the original annotation, requesting the official annotation from Zenodo maybe unnecessary.

**Note**: we did not provide the code to remove songs in test data from training data. Please implement it before running the preprocessing script `preprocess_dali_v2.py`.

Run the command below for preprocessing:

    python preprocess_dali_v2.py


### Hansen

1. Get the original dataset ready at `./raw/hansen`.
2. Run the command

        python preprocess_hansen.py

### Jamendo

1. Get the original dataset ready at `./raw/jamendo`.
2. Run the command
        
        python preprocess_jamendo.py

### Mauch

1. Get the original dataset ready at `./raw/mauch`.
2. Run the command
        
        python preprocess_mauch.py
