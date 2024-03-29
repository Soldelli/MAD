# MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions


## Introduction
Official repository for  MAD: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions. [[ArXiv Preprint](https://arxiv.org/abs/2112.00431)]

Paper accepted @CVPR22. 

<!-- <div align="center" valign="middle"><img height="250px" src="https://drive.google.com/uc?export=view&id=14c5sPpBLQlHLRXhZvM-61iieWDq0ZTbX"></div> -->
![mad](https://user-images.githubusercontent.com/26504816/144832743-a4852b5e-ec40-47e4-aa8c-b470e0638ef8.jpg)

# News
**[March 2023]** MAD-v2 is now available, check this [link](https://www.robots.ox.ac.uk/~vgg/research/autoad/) for more details. MADv2 refines the training annotations provided by MADv1 reducing transcription error and word recognition thanks to Whisper. Additionally, AutoAD introduces the first captioning baseline on MAD data. </br>
**[June 2022]** MAD-v1 accepted at CVPR2022. </br>
**[June 2022]** MAD-v1 release. </br>



# What is MAD?
MAD is a large-scale dataset collected from Movie Audio Descriptions for the Language Grounding in Videos task. It comprises a total of 384K sentences grounded in over 1.2K hours of continuous videos from 650 different and diverse movies. 

Spanning over 22 genres across 90 years of cinema history, MAD covers a broad domain of actions, locations, and scenes. Moreover, MAD inherits a diverse set of visual and linguistic content from the broad movie genres, ranging from fiction to everyday life. 

These properties yield a unique long-form grounding setup with a large linguistic vocabulary size and challenging requirements in terms of accuracy and efficiency. 



# Request access to the MAD dataset
To get access to the MAD dataset (annotations and pre-extracted features) please follow these steps:

1- Complete this form ([link](https://forms.gle/hxR4TrQPFuNGpzcr8)) and sign the NDA (Non Disclosure Agreement).

2- We will verify the correctness of the provided information. 

3- You will receive an email with the credentials to download the data.

See the data documentation [here](doc/README.md). 



# Data Download
The confirmation email will contain LINK and PASSWORD to access the data. Each file can be downloaded from the web interface or through the provided script `MAD_downloader.py`, which can be used as follows:

```bash 
python MAD_downloader.py --destination_folder {PATH_TO_DOWNLOAD} --download_link {LINK} --password {PASSWORD} 
```

Required Dependencies: `pip install google-measurement-protocol tqdm`



# Video-Language Grounding Baselines
This repository contains the code for the baselines used in the main paper. To reproduce the results and use our code follow these two links: 

- Zero Shot CLIP ([link](baselines/0ShotCLIP/))
- VLG-Net ([link](baselines/VLG-Net/))



# Support for additional visual/linguistic features
Due to copyright restrictions, we will not release the movies. 
Nonetheless, to foster new research and to keep up with constantly changing needs, the MAD team will provide support for the computation of visual/linguistic features upon request. </br>
If you are in need of a new set of features (besides those we already provide), please reach out via email or open an issue on this repo. </br>

We will soon release a docker image for simplifying this process.



# Citation
If any part of our paper is helpful to your work, please cite with:
```
@InProceedings{Soldan_2022_CVPR,
    author    = {Soldan, Mattia and Pardo, Alejandro and Alc\'azar, Juan Le\'on and Caba, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
    title     = {MAD: A Scalable Dataset for Language Grounding in Videos From Movie Audio Descriptions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5026-5035}
}

@article{rohrbach2017movie,
      title={Movie description},
      author={Rohrbach, Anna and Torabi, Atousa and Rohrbach, Marcus and Tandon, Niket and Pal, Christopher and Larochelle, Hugo and Courville, Aaron and Schiele, Bernt},
      journal={International Journal of Computer Vision},
      volume={123},
      number={1},
      pages={94--120},
      year={2017},
      publisher={Springer}
}
```

If you used MAD-v2 annotations kindly also cite this work:
```
@InProceedings{han2023autoad,
    title={{AutoAD}: Movie Description in Context},  
    author={Tengda Han and Max Bain and Arsha Nagrani and G\"ul Varol and Weidi Xie and Andrew Zisserman},  
    booktitle={CVPR},  
    year={2023}}
```
