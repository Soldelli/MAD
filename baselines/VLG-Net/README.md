# VLG-Net baseline for MAD dataset

## Introduction

This repo builds atop the official VLG-Net repository ([codebase](https://github.com/Soldelli/VLG-Net)). It is used to compute the baselines reported in the paper "MAD: A scalable dataset for language grounding in videos from movie audio descriptions" [[ArXiv Preprint](https://arxiv.org/abs/2112.00431)].

## Installation

Clone the repository and move to the folder:
```bash
https://github.com/Soldelli/MAD
cd MAD/baselines/VLG-Net/
```

Install environmnet:
```bash
conda env create -f environment.yml
```

If installation fails, please follow the instructions in file `doc/environment.md` [(link)](doc/environment.md).

## Data
Kindly get access to the data first. Follow the instructions on the main page of the repository ([link](https://github.com/Soldelli/MAD)). </br>
Once you obtain the data, follow the folder structure highlighted below. Also, download the following additional files:

| **Resource** | Download Link  | File Size | Destination Folder |
| ----         |:-----:         |:-----:    |  :-----:    |
| **GCNeXt Warmup**          |  [link](https://drive.google.com/file/d/1KLuKR_Wv1-wrAL1qyzN85XN4-GTCavHV/view?usp=sharing) | (~0.1GB) | `./datasets/`|
| **Pretrained Models**      |  [link](https://drive.google.com/file/d/1e6btL4LLgaogG9RKPg_l9-wfTWH__Qar/view?usp=sharing) | (~0.1GB) | `./models/`  |


The folder structure should be as follows:
```
.
├── datasets
│   ├── gcnext_warmup
│   └── MAD
│       ├── annotations
│       │   ├── MAD_train.json
│       │   ├── MAD_val.json
│       │   └── MAD_test.json
│       └── features
│           ├── CLIP_language_tokens_features.h5
│           └── CLIP_frames_features_5fps.h5
│
├── doc
│
├── lib
│   ├── config
│   ├── data
│   ├── engine
│   ├── modeling
│   ├── structures
│   └── utils
│
├── models
│   └── mad_best_model
│
├── outputs
│
└── scripts
```

## Training

Copy paste the following commands in the terminal. </br>

```bash
conda activate vlg
python train_net.py --config-file configs/mad.yml OUTPUT_DIR outputs/mad
```


## Evaluation
We provide scripts to automatically run the inference on pre-trained models. See script details if you want to run inference on a custom model. </br>

```bash
conda activate vlg
bash scripts/mad_reproduce_best.sh
```

## Citation
If any part of our paper and code is helpful to your work, please cite with:

```
@inproceedings{soldan2021mad,
  title={MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  author={Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Heilbron, Fabian Caba and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

@inproceedings{soldan2021vlg,
  title={VLG-Net: Video-language graph matching network for video grounding},
  author={Soldan, Mattia and Xu, Mengmeng and Qu, Sisi and Tegner, Jesper and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3224--3234},
  year={2021}
}
```