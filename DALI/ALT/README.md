# Automatic Lyric Transcription with DALI / Hansen / Jamendo / Mauch
This sub-project contains recipes for training audio-only ALT system for the DALI / Hansen / Jamendo / Mauch Dataset. We assume you have downloaded, pre-processed, and saved these datasets properly.

## Prerequisites
Please refer to `DALI/LM/README.md` to train and save language model before running following experiments. The trained RNNLM is saved at `/path/to/RNNLM`.

## How to run

1. Prepare DALI dataset, run:
```
python dali_prepare.py --train_folder /path/to/DALI_v2 --test_folder /path/to/DALI_Test
```

2. Prepare Hansen / Jamendo / Mauch three evaluation datasets, run:
```
python eval_prepare.py --hansen_folder /path/to/Hansen --jamendo_folder /path/to/Jamendo --mauch_folder /path/to/Mauch
```

The resulting organization for csv files folder should be like:
```
data
├── train.csv
├── valid.csv
├── test.csv
├── Hansen.csv
├── Jamendo.csv
├── Mauch.csv
```

3. Train the ALT system on DALI train split, validate it on DALI valid split, and evaluate it on DALI test split / Hansen / Jamendo / Mauch Dataset, run:
```
CUDA_VISIBLE_DEVICES=0,1 python train_wav2vec2_tb.py hparams/train_wav2vec2.yaml --data_parallel_backend --data_folder /path/to/DALI_v2 --pretrained_lm_path /path/to/RNNLM --pretrain True
```
The option `--pretrain` is used to load the pretrained model in the folder `DSing/save_model/`.

We use two A5000 GPUs (each has 23 GB) to run experiments. 


## Results
| Release | hyperparams file | DALI test WER | Hansen WER | Jamendo WER | Mauch WER | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| -----:| -----:| :-----------:|
| 22-10-04 | train_wav2vec2.yaml |  30.85 | 18.71 | 33.13 | 28.48 | https://drive.google.com/drive/folders/15e0IBLJHtTBpKfI0ju4EFR0dxLM63LW-?usp=sharing | 2xA5000 23GB |