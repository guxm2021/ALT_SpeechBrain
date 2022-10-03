# Language Model with DALI
This sub-project contains recipes for training RNN langugae model for the DALI Dataset. We assume you have downloaded and pre-processed DALI dataset and DSing dataset. The DALI dataset is saved at `/path/to/DALI_v2` and `/path/to/DALI_Test` while the DSing dataset is saved at `/path/to/DSing`

## Prerequisties
We train RNNLM on DSing and DALI training splits and evaluate the model on DALI valid split. Please follow the step 1-3 to prepare the data.

1. Prepare text corpus for DALI, run:
```
python dali_prepare.py --data_folder /path/to/DALI_v2
```

2. Prepare text corpus for DSing, run:
```
python dsing_prepare.py --data_folder /path/to/DSing
```

3. Combine text corpus from two datasets into train split, run:
```
python text_prepare.py
```

## How to run

Train the RNNLM for DALI dataset, run:
```
python train_rnnlm.py hparams/train_rnnlm.yaml
```
The results are saved to `results/RNNLM_mix/<seed>/CKPT-files`. We mark the CKPT folder of best model is `/path/to/RNNLM`. More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).


## Results
| Release | hyperparams file | Tokenizer | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| --------:| :-----------:|
| 22-10-01 | train_rnnlm.yaml |  Character | https://drive.google.com/drive/folders/1EPIWTCH8e4R8oki987PdqRjFN-A7nXP8?usp=sharing | 1xA5000 23GB |