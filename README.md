# ALT_SpeechBrain
The repository is an official PyTorch implementation of following papers: 

[MM-ALT: A Multimodal Automatic Lyric Transcription System](https://arxiv.org/abs/2207.06127) (ACM-MM 2022)
![image](assets/mmalt_framework.png)


[Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription](https://arxiv.org/abs/2207.09747) (ISMIR 2022)

![image](assets/transfer_framework.png)

## Introduction
MM-ALT is a multi-modal automatic lyric transcription framework accepting audio, video, and IMU modalities, which is proposed in our ACM-MM 2022 paper. The transfer learning techniques in MM-ALT are well-explored in our ISMIR 2022 paper. Through transfer learning, our audio-only ALT system can achieve state-of-the-art performance on multiple benchmark datasets, including DSing, DALI, Hansen, Jamado, and Mauch.

If you find this repo useful in your research, please consider citing our papers:
```BibTex
@article{gu2022mm,
  title={MM-ALT: A Multimodal Automatic Lyric Transcription System},
  author={Gu, Xiangming and Ou, Longshen and Ong, Danielle and Wang, Ye},
  journal={arXiv preprint arXiv:2207.06127},
  year={2022}
}

@article{ou2022towards,
  title={Towards Transfer Learning of wav2vec 2.0 for Automatic Lyric Transcription},
  author={Ou, Longshen and Gu, Xiangming and Wang, Ye},
  journal={arXiv preprint arXiv:2207.09747},
  year={2022}
}
```

## Installation
### Environement
Install Anaconda and create the environment with python 3.8.12, pytorch 1.9.0 and cuda 11.1:
```
conda create -n mmalt python=3.8.12
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### SpeechBrain

We run experiments based on [SpeechBrain toolkit](https://arxiv.org/pdf/2106.04624.pdf). For simiplicity, we remove the original recipes. To install SpeechBrain, run following commands:
```
cd MMALT
pip install -r requirements.txt
pip install --editable .
```

[Transformers](https://arxiv.org/pdf/1910.03771.pdf) and other packages are also required:
```
pip install transformers
pip install datasets
pip install sklearn
```

### AV-Hubert
We adapt [AV-Hubert (Audio-Visual Hidden Unit BERT)](https://arxiv.org/pdf/2201.02184.pdf) in our experiments. To enable the usage of AV-Hubert, run following commands:
```
cd ..
git clone https://github.com/facebookresearch/av_hubert.git
cd av_hubert
git submodule init
git submodule update
```

[Fairseq](https://arxiv.org/pdf/1904.01038.pdf) and other packages are also required:
```
pip install -r requirements.txt
cd fairseq
pip install --editable ./
```

## Training and Evaluation
The code will be released soon.

## Acknowledgement
Part of the code is borrowed from [SpeechBrain](https://github.com/speechbrain/speechbrain) and [AV-Hubert](https://github.com/facebookresearch/av_hubert).