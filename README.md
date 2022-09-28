# Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription
This is the author's official PyTorch implementation for ALT_SpeechBrain. This repo contains code for experiments in the **ISMIR 2022** paper:

[Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription](https://guxm2021.github.io/guxm.github.io/pdf/ISMIR2022.pdf)


## Project Description
Automatic speech recognition (ASR) has progressed significantly in recent years due to the emergence of large-scale datasets and the self-supervised learning (SSL) paradigm. However, as its counterpart problem in the singing domain, the development of automatic lyric transcription (ALT) suffers from limited data and degraded intelligibility of sung lyrics. To fill in the performance gap between ALT and ASR, we attempt to exploit the similarities between speech and singing. In this work, we propose a transfer-learning-based ALT solution that takes advantage of these similarities by adapting wav2vec 2.0, an SSL ASR model, to the singing domain. We maximize the effectiveness of transfer learning by exploring the influence of different transfer starting points. We further enhance the performance by extending the original CTC model to a hybrid CTC/attention model. Our method surpasses previous approaches by a large margin on various ALT benchmark datasets. Further experiments show that, with even a tiny proportion of training data, our method still achieves competitive performance.

## Method Overview
<p align="center">
<img src="assets/transfer_framework.png" alt="" data-canonical-src="assets/transfer_framework.png" width="100%"/>
</p>

## Installation
### Environement
Install Anaconda and create the environment with python 3.8.12, pytorch 1.9.0 and cuda 11.1:
```
conda create -n alt python=3.8.12
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### SpeechBrain

We run experiments based on [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain). For simiplicity, we remove the original recipes. To install SpeechBrain, run following commands:
```
cd MM_ALT
pip install -r requirements.txt
pip install --editable .
```

[Transformers](https://github.com/huggingface/transformers) and other packages are also required:
```
pip install transformers
pip install datasets
pip install sklearn
```

## Training and Evaluation
The code will be released soon.

## Citation
```BibTex
@article{ou2022towards,
  title={Towards Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription},
  author={Ou, Longshen and Gu, Xiangming and Wang, Ye},
  journal={arXiv preprint arXiv:2207.09747},
  year={2022}
}
```
We borrow the code from [AV-Hubert](https://arxiv.org/pdf/2201.02184.pdf), please also consider citing their works.


## Also Check Our Relevant Work
**MM-ALT: A Multimodal Automatic Lyric Transcription System**<br>
Xiangming Gu*, Longshen Ou*, Danielle Ong, Ye Wang<br>
*ACM International Conference on Multimedia (ACM MM), 2022, (Oral)*<br>
[[paper](https://guxm2021.github.io/guxm.github.io/pdf/ACMMM2022.pdf)][[code](https://github.com/guxm2021/MM_ALT)]


## License
ALT_SpeechBrain is released under the Apache License, version 2.0.
