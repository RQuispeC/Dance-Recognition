import bag_of_words
import dance_detection_filter_faces

import cv2
import argparse as ap
import numpy as np
import os

from sklearn.externals import joblib
from sklearn import svm
def recoverClassificationTrainedModels(featureType, n_word, clustering_method, type_pooling, type_coding):
  models_data = featureType + '_' + str(n_word) + '_' + clustering_method + '_' + type_pooling + '_' + type_coding
  codebook = joblib.load(models_data + '_codebook.pkl')
  svm = joblib.load(models_data + '_svm.pkl')
  return svm, codebook

if __name__ == '__main__':
  # mode defines the type of results, 
  # debug mode saves plot of each step 
  # test mode just saves plot of body detection.
  mode = 'debug' #test
  
  # set path to data
  images_path = ''
  

  hog_files_path = ''

  # =================================== DETECT DANCER ==============================================
  for   


  # =================================== CLASSIFY DANCER ==============================================
  #set parameter to recover trained models
  featureType = 'SIFT' #SURF, SIFT
  n_word = 320
  type_coding = 'hard'
  type_pooling = 'max'
  clustering_method = 'random'

  svm, codebook = recoverClassificationTrainedModels(featureType, n_word, clustering_method, type_pooling, type_coding)



  



