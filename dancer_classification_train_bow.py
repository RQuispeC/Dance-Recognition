import bag_of_words
import cv2
import argparse as ap
import numpy as np
import os

from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn import cross_validation
from sklearn import svm, grid_search, metrics
import time

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

def run_gridSearch(classifier, args, data_training, label_training, data_validation, label_validation):
    from sklearn import metrics
    return metrics.accuracy_score(classifier(**args).fit(data_training, label_training).predict(data_validation), label_validation)

def readRectangle(file_path):
  file = open(file_path, "r")
  for line in file:
    tmp = line.replace('"', '').split()
    return (int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3]))  # list of strings

def stackFeatures(features):
  descriptors = features[0][1]
  for descriptor in features[1:]:
    descriptors = np.vstack((descriptors, descriptor))
  return descriptors

def getData(train_path, featureType = 'SURF'):
  # Get the training classes names and store them in a list
  training_names = os.listdir(train_path)
  training_names.sort(reverse = True)

  labels = []
  features = []
  K = 0

  for it in xrange(len(training_names)):
    training_name = training_names[it]
    images_names = os.listdir(train_path + training_name)
    images_names.sort(reverse = True)

    print '\tfor ', training_name
    for image_name in images_names:
      extension = image_name[image_name.rfind('.'):]
      if extension == '.txt':
        continue

      img = cv2.imread(train_path + training_name + '/' + image_name)
      rectangle = readRectangle(train_path + training_name + '/' + image_name[:image_name.rfind('.')] + '.txt')
      print rectangle
      img = img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]

      if featureType == 'SURF':
        _, feature = surf.detectAndCompute(img, None)
        K = feature.shape[1]
      elif featureType == 'SIFT':
        _, feature = sift.detectAndCompute(img, None)
        K = feature.shape[1]

      
      labels.append(it)
      features.append(feature)

    print '\tdone'  

  return features, labels, training_names, K

def getHistograms(data_train, stacked_data, N, K, c_method = 'random', n_words = 300 , type_coding = 'hard', type_pooling = 'sum', save_data = ''):
  # Let's suppose that you have N images, each image contains 1..M data-points, each data-point has a length of K features
  # data_train will contain all this information, so that, it will have a shape of NxMxK (however, the second dimension is not fixed since it can vary, but just suppose that it is constant)
  begin = time.time()
  print '\tBuilding codebook ...'
  # To build the codebook, in theory, we must use all data-points, so that:
  words, codebook = bag_of_words.build_codebook(
    data_train = stacked_data.reshape(-1, K),
    number_of_words = n_words, 
    clustering_method = c_method
  )
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin

  print '\tSaving codebook ...'
  joblib.dump(codebook, save_data + '_codebook.pkl')
  print '\tdone'
  
  # Once the codebook is built, the bag-of-words histograms can be calculated through:
  all_features = np.empty([N, n_words])

  # If you want to use multi-processing
  begin = time.time()
  print '\tGetting Histograms ...'
  all_features = Parallel(n_jobs = -1) (delayed(bag_of_words.coding_pooling_per_video)(
    codebook_predictor = codebook, 
    number_of_words = n_words, 
    data_per_video = np.array(data_image).reshape(-1, K), 
    type_coding = type_coding, 
    type_pooling = type_pooling,) 
    for data_image in data_train)
  print '\tdone ...'
  end = time.time()
  print 'Elapset time:', end - begin

  
  all_features = np.array(all_features).reshape(-1, n_words)
  
  return all_features

def runClassification(img_features, img_labels, featureType, save_data):
  
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(img_features, np.array(img_labels), test_size=0.2, random_state=0)
  gamma_range = np.power (10., np.arange (-5, 5, 0.5));
  C_range = np.power (10., np.arange (-5, 5));
  grid_search_params = \
    [{'kernel' : ['rbf'], 'C' : C_range, 'gamma' : gamma_range},\
     {'kernel' : ['linear'], 'C' : C_range}];
    
  classifier = svm.SVC
  
  begin = time.time()
  print '\trunning grid search ...'
  grid_search_ans = Parallel(n_jobs = -1)(delayed(run_gridSearch)(classifier, args, X_train, y_train, X_test, y_test) for args in list(grid_search.ParameterGrid(grid_search_params)))
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin


  begin = time.time()
  print '\tgetting best parameters ...'
  best_params = list(grid_search.ParameterGrid(grid_search_params))[grid_search_ans.index(max(grid_search_ans))]
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin


  begin = time.time()
  print '\ttraining SVM ...'  
  clf = classifier(**best_params).fit(X_train, y_train)
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin


  begin = time.time()
  print '\tsaving model ...'
  joblib.dump(clf, save_data + '_svm.pkl')
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin


  begin = time.time()
  print '\trunning test ...'
  pred = clf.predict(X_test)

  print metrics.classification_report(pred, y_test)
    
  print '\taccuracy: ', metrics.accuracy_score(pred, y_test)
  print '\tdone'
  end = time.time()
  print 'Elapset time:', end - begin

  report = open(save_data + '.txt', 'w')
  report.write(save_data)
  report.write(str(metrics.classification_report(pred, y_test)))
  report.write('accuracy: ' + str (metrics.accuracy_score(pred, y_test)))
  report.close()


  #print 'bow_' + featureType + '_' + '_'.join(np.array([[k, v] for k, v in zip(best_params.keys(), best_params.values())]).flatten()) + '.pkl'
  print save_data

if __name__ == '__main__':
  # Get the path of the training set
  parser = ap.ArgumentParser()
  parser.add_argument("-d", "--dataset", help="Path to dataset", required="True")
  args = vars(parser.parse_args())

  featureType = 'SIFT' #SURF, SIFT
  #n_word = 320
  type_coding = 'hard'
  #type_pooling = 'max'
  clustering_method = 'random'

  #read images to get features and labels
  #features is a list of the featureType descriptor/features 
  print 'Reading data and getting', featureType, 'features ...'
  features, labels, classNames, K = getData(args["dataset"], featureType)
  print 'done'
  stacked_data = stackFeatures(features)

  for type_pooling in ['mean', 'sum', 'max', 'min']:
    for n_word in [1200, 1000, 800, 550, 420, 320, 220, 120, 70, 50]:  
      #get histograms
      save_data = featureType + '_' + str(n_word) + '_' + clustering_method + '_' + type_pooling + '_' + type_coding  
      print 'Getting histograms ...'
      histograms = getHistograms(features, stacked_data, len(features), K, clustering_method, n_word, type_coding, type_pooling, save_data)
      print 'done' 

      #run classification
      print 'Running classification ...'
      runClassification(histograms, labels, featureType, save_data)
      print 'done'
      print '+++++++++++++++++++++++++++++++++++++++++++++'

