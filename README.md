### Disclaimer
Please note that this readme is out of date, so it may not include full instructions. Author may update it in the future.

# Mask Detection and Dancers Recognition

This project aims at detecting mask location and recognize images for typical dances from the city of Cusco - Peru.

Basic system requirements:
- OpenCV  3.0.0 or greater
- Numpy 1.11.2               
- Scikit-image 0.12.3
- Scikit-learn 0.18

##  Getting the labels

**getLabels.py:** Divides the original images into small pieces and labels them with the percentage of the containing face. The dataset file must contain `~/Path-to-original-dataset/rgb` and `~/Path-to-original-dataset/bin` subdirectories. Put original images in `~/Path-to-original-dataset/rgb` and the binary images (ground truth) in `~/Path-to-original-dataset/bin`. The pieces will be saved in a `/patches` directory at same hierarchy
of `~/Path-to-original-dataset`, a subdirectory for each type of `w, h, pyr_hight` and `shift` is created. 

To execute the script, run:

`python getLabels.py -d ~/Path-to-original-dataset`

## Training the models

**train.py:** Uses the results of `getLabels.py` to train a model (SVM, Random Forest, AdaBoost) and save the learned parameters in `/models` directory at same hierarchy of the `~/patches` directory. A subdirectory for each `w, h, pyr_hight, shift` and `descriptor parameters` is created. 

Run it with:

`python train.py -d ~/patches/directory-with-small-labeled-images`

##Running a test

**detect.py:** Reads the trained models, and for each image in the directory plots an image with the face locations. 

Run it with:

`python detect.py -m ~/models/directory-with-learned-models  -t ~/path-to-test-directory`

