# Language Model with DSing
This sub-project contains recipes for training RNN langugae model for the DSing Dataset. We assume you have downloaded and pre-processed DSing dataset. The DSing dataset is saved at `/path/to/DSing`.

## How to run

1. Prepare Text corpora, run:
```
python dsing_prepare.py --data_folder /path/to/DSing
```

2. Train the RNNLM for DSing dataset, run:
```
python train_rnnlm.py hparams/train_rnnlm.yaml --duration_threshold 28
```
The results are saved to `results/RNNLM_duration28/<seed>/CKPT-files`. We mark the CKPT folder of best model is `/path/to/RNNLM`. More details about how to save and use CKPT files are in [SpeechBrain Toolkit](https://speechbrain.github.io).


## Results
| Release | hyperparams file | Tokenizer | Val. loss | Model link | GPUs |
|:-------------:|:---------------------------:| -----:| -----:| --------:| :-----------:|
| 22-09-30 | train_rnnlm.yaml |  Character | 0.775 | https://drive.google.com/drive/folders/1lt72ZznNAaHxjNtBbHPbyoxfYHWlWogY?usp=sharing | 1xA5000 23GB |