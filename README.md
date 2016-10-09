# Face Detector

This project aims to detect face location in images of typical dance, to run it you need:
- OpenCV  3.0.0 or greater
- Numpy 1.11.2               
- Skimage 0.12.3
- SkLearn 0.18

##  Getting the labels

getLabels.py: Divides the original images into small pieces and labeled with the percentage of face containing, 
the dataset file must contain `~/Path-to-original-dataset/rgb` and `~/Path-to-original-dataset/bin` subdirectories, put original images in `~/Path-to-original-dataset/rgb` and binary images (face 
parts with black and white otherwise) in `~/Path-to-original-dataset/bin`. The pieces will be saved in a `/patches` directory at same hierarchy
of the `~/Path-to-original-dataset` directory, a subdirectory for each type of `w, h, pyr_hight` and `shift` is created. Run it with:


`python getLabels.py -d ~/Path-to-original-dataset`

## Training the models

train.py: Uses the result of getLabels.py to train a model (SVM, Random Forest, AdaBoost) and save the learned 
parameters in /models directory at same hierarchy of the `~/patches` directory, a subdirectory for each type of 
`w, h, pyr_hight, shift` and `descriptor parameters` is created. Run it with:

`python train.py -d ~/patches/directory-with-small-labeled-images`

##Running a test

detect.py: Reads the trained models, and for each image in the directory plots and image with the face locations. Run it with:

`python detect.py -m ~/models/directory-with-learned-models  -t ~/path-to-test-directory`

