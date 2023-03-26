# MAD Data documentation
This document contains the up-to-data list of data files released by MAD. 
If you do not have access to the data yet, please follow the steps [here](../README.md/#Request-access-to-the-MAD-dataset).

## Data ZOO:
- `DataInspection.ipynb`: A jupyter notebook containing the data walkthrough. It showcases how to navigate the content of all the files below. 

- `annotations.tar.xz`: Tar file containing the annotations for the dataset. It contains a single json file for each split. 

- `CLIP_B32_frames_features_5fps.h5`: H5 file containing the frame-level embeddings for the videos contained in MAD computed with the CLIP B/32 model. Frames are extracted at 5FPS. It is used for computing the performance of Zero-Shot CLIP and VLG-Net baselines. 

- `CLIP_B32_language_tokens_features.h5`: H5 file containing the token-level embeddings for the entire dataset's query annotations extracted with the CLIP B/32 model. This file is used to train and evaluate the VLG-Net baseline. 

- `CLIP_B32_language_features_MAD_test.h5`: H5 file containing the sentence-level embeddings for the test set query annotations extracted with the CLIP B/32 model. This file is only used to compute the Zero-Shot CLIP baseline performance. 

- `CLIP_L14_frames_features_5fps.h5`: H5 file containing the frame-level embeddings for the videos contained in MAD computed with the CLIP L/14 model. Frames are extracted at 5FPS. 

- `CLIP_L14_language_tokens_features.h5`: H5 file containing the token-level embeddings for the entire dataset's query annotations extracted with the CLIP L/14 model. 

