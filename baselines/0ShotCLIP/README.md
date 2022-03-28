# CLIP based zero-shot baseline for MAD dataset

## Introduction

This repo includes the code for the zero-shot baseline computed atop CLIP [[Paper](https://arxiv.org/abs/2103.00020), [Code](https://github.com/openai/CLIP)] embeddings for the paper "MAD: A scalable dataset for language grounding in videos from movie audio descriptions" [[ArXiv Preprint](https://arxiv.org/abs/2112.00431)].

## Installation

Clone the repository and move to the folder:
```bash
https://github.com/Soldelli/MAD
cd MAD/baselines/0ShotClip/
```

Install environmnet:
```bash
conda env create -f environment.yml
```

If installation fails, please follow the instructions in file `doc/environment.md` [(link)](doc/environment.md).

## Data
Kindly get access to the data first. Follow the instructions on the main page of the repository ([link](https://github.com/Soldelli/MAD)). </br>
Once you obtain the data, follow the folder structure highlighted below. Place the files in the correct folders and proceed with the rest of the README. 
</br>

The folder structure should be as follows:
```
.
├── datasets
│   └── MAD
│       ├── annotations
│       │   ├── MAD_train.json
│       │   ├── MAD_val.json
│       │   └── MAD_test.json
│       └── features
│           ├── CLIP_language_features_MAD_test.h5
│           └── CLIP_frames_features_5fps.h5
└── doc
```

## Evaluation
Simply run the command: </br>
```bash
conda activate 0ShotCLIP
python 0_shot_baseline.py
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

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```