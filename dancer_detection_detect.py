import argparse as ap
import cv2
import numpy as np
import os
import re as regex

from sklearn import svm, ensemble
from skimage.feature import hog
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.externals import joblib
import pylab as plt
from skimage.filters import sobel
from sklearn.externals.joblib import Parallel, delayed

def loadModels():
    clf1 = joblib.load('/home/rodolfo/Pictures/dances-data/ds4/models/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_RandomForest_n_estimators_150/' + 'RandomForestClassifier.pkl')

    clf2 = joblib.load('/home/rodolfo/Pictures/dances-data/ds4/models/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_SVM_C_100_gamma_0.001/'+ 'SVMClassifier.pkl')
    

    return clf1, clf2

def recoverMetadata(models_path):
    models_path =  os.path.basename(os.path.normpath(models_path))
    myMetadata = models_path.split('_')
    return myMetadata[4] == "sobel", int(myMetadata[0]), int(myMetadata[1]), int(myMetadata[2]), tuple(map(int, regex.findall(r'\d+', myMetadata[8]))), tuple(map(int, regex.findall(r'\d+', myMetadata[9]))), int(myMetadata[7])

def createDefaultDirectory(models_path):

    directory1 = os.path.join('/home/rodolfo/Pictures/dances-data/ds4/', "results/" + '3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_RandomForest_n_estimators_150/')
    print "Data will be saved in: ", directory1
    if not os.path.exists(directory1):
            os.makedirs(directory1)

    directory2 = os.path.join('/home/rodolfo/Pictures/dances-data/ds4/', "results/" + '3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_SVM_C_100_gamma_0.001/')
    print "Data will be saved in: ", directory2
    if not os.path.exists(directory2):
            os.makedirs(directory2)

    return directory1, directory2

def createDirectory(saveDirectory, models_path):
    dirName =os.path.basename(os.path.normpath(models_path))
    directory = os.path.join(saveDirectory, "results/" + dirName)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
            os.makedirs(directory)
    return directory

def runTest(randomForestModel, svmModel, test_path, training_names, saveDirectory1 = "", saveDirectory2 = "", useSobel = True, pyr_hight = 3, h = 50, w = 50, params_hog = dict(orientations = 9, pixels_per_cell = (9, 9), cells_per_block = (2, 2))):
    print "Parameters are: "
    print useSobel, pyr_hight, h, w, params_hog.values()
    pyr_hight = 2
    training_names.sort(reverse = True)
    #training_names.sort()

    for name in training_names:
        extension = name[name.rfind('.'):]
        if extension == '.txt':
            continue

        image_path = os.path.join(test_path, name)
        if(useSobel):
            im_or = cv2.imread(image_path, False)
            im_or = sobel(im_or)
            im_or = cv2.normalize(im_or, None, 0, 255, cv2.NORM_MINMAX)
        else:
            im_or = cv2.imread(image_path)
            im_or = cv2.cvtColor(im_or, cv2.COLOR_BGR2GRAY)
        print name
        for k in range(pyr_hight):
            print "Factor : ", k+1
            if os.path.isfile(os.path.join(saveDirectory1, name[:name.rfind(".")] + "_" + str(k+1) + ".npz")):
                print 'already calculated'
                continue
        
            if os.path.isfile(os.path.join(saveDirectory2, name[:name.rfind(".")] + "_" + str(k+1) + ".npz")):
                print 'already calculated'
                continue

            im = cv2.resize(im_or, (0,0), fx = 1.0/(k+1.0), fy = 1.0/(k+1.0))
            n, m = im.shape[:2]
            if(n<h or m<w):
                continue
            detectImageclfRandom = np.zeros((n, m))
            detectImageclfSVM = np.zeros((n, m))
            for i in range(0, n-h, shift):
                for j in range(0, m-w, shift):
                    cropped = im[i:i+h, j:j+w]
                    features = hog(cropped, **params_hog)
                    features = features.reshape(1, len(features)) #reshape to 2D array
                    prediction1 = randomForestModel.predict(features)
                    prediction2 = svmModel.predict(features)
                    if((i*j)%50000 == 0 and i>0 and j > 0 ):
                        print "Detecting face"
                    detectImageclfRandom[i:i+h, j:j+w]+=prediction1
                    detectImageclfSVM[i:i+h, j:j+w]+=prediction2

            np.savez(os.path.join(saveDirectory1, name[:name.rfind(".")] + "_" + str(k+1) + ".npz"), detectImageclfRandom)
            np.savez(os.path.join(saveDirectory2, name[:name.rfind(".")] + "_" + str(k+1) + ".npz"), detectImageclfSVM)
 

if __name__ == '__main__':

    # Get the path of the trained models and test dataset
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--testingSet", help="Path to test dataset", required="True")
    args = vars(parser.parse_args())

    # Get the training path to labeled dataset
    test_path = args["testingSet"]
    training_names = os.listdir(test_path)
    models_path = '/home/rodolfo/Pictures/dances-data/ds4/models/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_RandomForest_n_estimators_150/'
    

    randomForestModel, svmModel = loadModels()

    #recover training metadata
    useSobel, pyr_hight, h, w, hogCellsPerBlock,  hogPixelPerCell, hogOrientation = recoverMetadata(models_path)
    shift = 5 #shift of sliding windows in test

    #create directory file for results
    directory1, directory2  = createDefaultDirectory(models_path)

    #process images
    runTest(randomForestModel, svmModel, test_path, training_names, directory1, directory2, useSobel, pyr_hight, h, w, dict(orientations = hogOrientation, pixels_per_cell = hogPixelPerCell, cells_per_block = hogCellsPerBlock))
