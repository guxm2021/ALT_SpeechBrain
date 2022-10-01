# Language Model with DALI
This sub-project contains recipes for training RNN langugae model for the DALI Dataset. We assume you have downloaded and pre-processed DSing dataset. The DSing dataset is saved at `/path/to/DALI`.

## How to run

1. Prepare Text corpora. We mix lyrics from the training set of DSing and the training set of DALI to train our RNN language model. For your convenience, we have organized the text into `data/train.txt`. The valid set of DALI has been saved to `data/valid.txt`

2. Train the RNNLM for DALI dataset, run:
```
python train_rnnlm.py hparams/train_rnnlm.yaml
```
The results are saved to `results/RNNLM_mix/<seed>/CKPT-files`. We mark the CKPT folder of best model is `/path/to/RNNLM`. More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).


## Results
| Release | hyperparams file | Tokenizer | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| --------:| :-----------:|
| 22-10-01 | train_rnnlm.yaml |  Character | https://drive.google.com/drive/folders/1EPIWTCH8e4R8oki987PdqRjFN-A7nXP8?usp=sharing | 1xA5000 23GB |