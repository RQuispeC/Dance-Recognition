import bag_of_words
import dancer_detection_filter_faces as face_detection

import cv2
import argparse as ap
import numpy as np
import os

from sklearn.externals import joblib
from sklearn import svm
from sklearn.externals.joblib import Parallel, delayed
import pylab as plt
import gc


sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

def plot_rectangles(img, rectangles, plotColor = (150, 255, 150)):
  for rectangle in rectangles:
    cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), plotColor, 2)
  return img

def plot_results(img , data , image_name, sufix = '', type = 'hasBoxes'):
  fig = plt.figure(figsize=(30, 20))
  plt.axis("off")
  if type == 'hog':
    a=fig.add_subplot(1,3,1)
    a.set_title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    a=fig.add_subplot(1,3,2)
    a.set_title("Escala 1")
    plt.imshow(data[0])

    a=fig.add_subplot(1,3,3)
    a.set_title("Escala 2")
    plt.imshow(data[1])

  if type == 'hasBoxes':
    a=fig.add_subplot(1,2,1)
    a.set_title("Escala 1")
    plt.imshow(cv2.cvtColor(plot_rectangles(img.copy(), data[0], (10, 200, 200)), cv2.COLOR_BGR2RGB))

    a=fig.add_subplot(1,2,2)
    a.set_title("Escala 2")
    plt.imshow(cv2.cvtColor(plot_rectangles(img.copy(), data[1], (200, 10, 200)), cv2.COLOR_BGR2RGB))
    
  if type == 'joint':
    img = plot_rectangles(img.copy(), data[0], (10, 200, 200))
    img = plot_rectangles(img.copy(), data[1], (200, 10, 200))
    a=fig.add_subplot(1,1,1)    
    a.set_title("faces")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  if type == 'body':
    max_area = 0
    max_ind = 0
    for ind  in xrange(len(data)):
      rectangle = data[ind]
      area = (rectangle[3] - rectangle[1]) *  (rectangle[3] - rectangle[0])
      if area > max_area:
        max_area = area
        max_ind = ind
    img = plot_rectangles(img.copy(), data, (10, 200, 200))
    img = plot_rectangles(img.copy(), [data[max_ind]], (200, 10, 200))

    a=fig.add_subplot(1,1,1)    
    a.set_title("faces")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  fig.savefig(os.path.join('/home/rodolfo/Pictures/', image_name + sufix+'.jpg'), bbox_inches='tight')

  #free ram 
  fig.clf()
  plt.close()
  del a
  gc.collect()

def hog_pyramid(img, hog_files_path, mode, image_name):
  dfs_threshold = [0.70, 0.50, 0.10]
  n, m = img.shape[:2]
  detected_hog = []
  detected_rectangles = []

  for ind in range(1, 3):
    #recover hog results
    print 'Current pyramid level:   ', ind
    file = np.load(os.path.join(hog_files_path, image_name+'_' + str(ind) + '.npz'))
    detectedClf = cv2.resize(file['arr_0'], (m, n))
    detected_hog.append(detectedClf)

    minPixelSize = 1000
    threshold = int((dfs_threshold[ind-1])*detectedClf.max())
    visited = np.zeros(detectedClf.shape[:2])
    rectanglesClf = face_detection.runImageBFS(threshold, detectedClf, minPixelSize, n, m)
    detected_rectangles.append(rectanglesClf)
  
  if mode == 'debug':
   plot_results(img, detected_hog, image_name, '_1_hog', type = 'hog')
   plot_results(img, detected_rectangles, image_name, '_2_hogRectangles', type = 'hasBoxes')

  return detected_rectangles

def face_refinement(img, hog_results, mode, image_name):
  faces_all_scales = []
  for ind in range(1, 3):
    faces = face_detection.improveFaces(img.copy(), hog_results[ind - 1])
    faces_all_scales.append(faces)
  
  if mode == 'debug':
   plot_results(img, faces_all_scales, image_name, '_3_refinement', type = 'hasBoxes')

  return faces_all_scales

def false_positives_supression(img, masks_all_scales, mode, image_name):
  detected_faces = []
  for masks in masks_all_scales:
    masks = face_detection.filterFalsePositives(img, masks, image_name)
    detected_faces.append(masks)
  
  if mode == 'debug':
   plot_results(img, detected_faces, image_name, '_4_faceDetection', type = 'hasBoxes')
   plot_results(img, detected_faces, image_name, '_5_faceDetection', type = 'joint')

  return detected_faces

def bodyEstimation(img, dancer_masks, mode, image_name):
  dancer_masks[0].extend(dancer_masks[1])

  faces = face_detection.non_max_suppression_fast(np.array(dancer_masks[0]), 0.80)
  bodyPercentage = 0.75
  body = face_detection.extentToBody(img, faces, bodyPercentage)
  
  if mode == 'debug':
   plot_results(img.copy(), body, image_name, '_6_body', type = 'body')

  body = face_detection.compressBody(img, body)

  if len(body) == 0:
    print '============has no body============='
    body = [(0, 0, img.shape[1], img.shape[0])]
  
  plot_results(img.copy(), body, image_name, '_7_body', type = 'body')
  
  return img[body[0][1]:body[0][3], body[0][0]:body[0][2]]

def detectDancer(images_path, image_name, hog_files_path, mode):
  name = image_name[:image_name.rfind('.')]
  img = cv2.imread(os.path.join(images_path, image_name))
  # =========== hog pyramid ==============
  print 'HOG Pyramid ...'
  hog_results = hog_pyramid(img, hog_files_path, mode, name)
  print 'hecho!!'

  # =========== face detection refinemnet ==============
  print 'Face Detection Refinement ...'
  masks = face_refinement(img, hog_results, mode, name)
  print 'hecho!!'

  # =========== false positives supression ==============
  print 'False Positives Supression ...'
  dancer_masks = false_positives_supression(img, masks, mode, name)
  print 'hecho!!'

  # =========== body estimation ==============
  print 'Body Estimation ...'
  dancer = bodyEstimation(img, dancer_masks, mode, name)
  print 'hecho!!'

  return dancer

def loadModels(featureType, n_word, clustering_method, type_pooling, type_coding):
  models_data = featureType + '_' + str(n_word) + '_' + clustering_method + '_' + type_pooling + '_' + type_coding
  codebook = joblib.load('models/'+ models_data + '_codebook.pkl')
  svm_classifier = joblib.load('models/'+ models_data + '_svm.pkl')
  return svm_classifier, codebook

def classifyDancer(img, svm_classifier, codebook, featureType, n_word, clustering_method, type_pooling, type_coding):
  print 'Getting Features ...'
  if featureType == 'SURF':
    _, feature = surf.detectAndCompute(img, None)
    K = feature.shape[1]
  elif featureType == 'SIFT':
    _, feature = sift.detectAndCompute(img, None)
    K = feature.shape[1]
  print 'hecho!!'


  print 'Getting Histogram ...'
  histogram = np.empty(n_word)

  histogram = bag_of_words.coding_pooling_per_video(
    codebook_predictor = codebook,
    number_of_words = n_word,
    data_per_video = feature.reshape(-1, K),
    type_coding = type_coding, #type_coding = hard
    type_pooling = type_pooling, #type_pooling = sum
  )

  histogram = np.array(histogram).reshape(-1, n_word)
  print 'hecho!!'  

  print 'Predicting ...'
  pred = svm_classifier.predict(histogram)
  print 'hecho!!'  

  return pred[0]

if __name__ == '__main__':
  featureType = 'SIFT' #SURF, SIFT
  n_word = 1000
  type_coding = 'hard'
  type_pooling = 'sum'
  clustering_method = 'random'

  dances = ['NEGRILLO', 'CONTRADANZA', 'QHAPAC COLLA', 'QHAPAC CHUNCHO']
  print 'Cargando modelos ...'
  svm_classifier, codebook = loadModels(featureType, n_word, clustering_method, type_pooling, type_coding)
  print 'hecho'

  # mode defines the type of results, 
  # debug mode saves plot of each step 
  # test mode just saves plot of body detection.
  mode = 'debug' #test
  
  # set path to data
  images_path = '/home/rodolfo/Pictures/demo-data/images/' 

  hog_files_path = '/home/rodolfo/Pictures/demo-data/hog-files/'


  image_names = os.listdir(images_path)
  image_names.sort()

  for image_name in image_names:
    extension = image_name[image_name.rfind('.'):]
    if extension == '.txt':
      continue

    print 'Imagen :' , image_name
    # =================================== DETECT DANCER ==============================================
    dancer = detectDancer(images_path, image_name, hog_files_path, mode)
    # =================================== CLASSIFY DANCER ==============================================
    dance_label = classifyDancer(dancer, svm_classifier, codebook, featureType, n_word, clustering_method, type_pooling, type_coding)

    print 'Danza:', dances[dance_label]
    #raw_input ("Presione ENTER para continuar ...")
