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

def loadModels(models_path, classifier='SVC'):
    # classifier can be a string or a list, in case of a string it returns one classifier, in case of
    # list it returns a list of the corresponding models
    if classifier.__type__ == 'str':
      classifier = [classifier]
    clf = []
    for clf_name in classifier:
      print "Loading " + clf_name
      clf.append(joblib.load(os.path.join(models_path, clf_name + ".pkl")))
      print clf_name + " loaded successfully"

    if len(clf) == 1:
      return clf[0]
    return clf

def recoverMetadata(models_path):
    models_path =  os.path.basename(os.path.normpath(models_path))
    myMetadata = models_path.split('_')
    return myMetadata[4] == "sobel", int(myMetadata[0]), int(myMetadata[1]), int(myMetadata[2]), tuple(map(int, regex.findall(r'\d+', myMetadata[8]))), tuple(map(int, regex.findall(r'\d+', myMetadata[9]))), int(myMetadata[7])

def runTest(clf, reg, test_path, training_names, useSobel = True, pyr_hight = 3, h = 50, w = 50, hogPixelPerCell = (9, 9), hogCellsPerBlock  = (2, 2), hogOrientation = 9):
    print "Parameters are: "
    print useSobel, pyr_hight, h, w, hogPixelPerCell, hogCellsPerBlock, hogOrientation

    for name in training_names:
        image_path = os.path.join(test_path, name)
        im_or = cv2.imread(image_path, False)
        if(useSobel):
            im_or = sobel(im_or)
            im_or = cv2.normalize(im_or, None, 0, 255, cv2.NORM_MINMAX)
        else:
            im_or = cv2.cvtColor(im_or, cv2.COLOR_BGR2GRAY)
        print name
        for k in range(pyr_hight):
            im = cv2.resize(im_or, (0,0), fx = 1.0/(k+1.0), fy = 1.0/(k+1.0))
            print "Factor : ", k+1
            n, m = im.shape[:2]
            if(n<h or m<w):
                continue
            detectImageclf = np.zeros((n, m))
            detectImagereg = np.zeros((n, m))
            #detectImage = np.zeros((n, m))
            for i in range(0, n-h, shift):
                for j in range(0, m-w, shift):
                    cropped = im[i:i+h, j:j+w]
                    features = hog(cropped, orientations=hogOrientation,  pixels_per_cell=hogPixelPerCell, cells_per_block=hogCellsPerBlock, visualise=False)
                    features = features.reshape(1, len(features)) #reshape to 2D array
                    prediction = clf.predict(features)
                    #detectImage[i, j] = prediction
                    regression = reg.predict(features)
                    if((i*j)%50000 == 0 and i>0 and j > 0 ):
                        print "Detecting face"
                    detectImageclf[i:i+h, j:j+w]+=prediction
                    detectImagereg[i:i+h, j:j+w]+=regression
            fig = plt.figure()
            a=fig.add_subplot(1,3,1)
            a.set_title('Original')
            plt.imshow(im, cmap='Greys_r')

            a=fig.add_subplot(1,3,2)
            a.set_title('Clasificador')
            plt.imshow(detectImageclf)

            a=fig.add_subplot(1,3,3)
            a.set_title('Regresor')
            plt.imshow(detectImagereg)
            '''
            a=fig.add_subplot(1,4,4)
            a.set_title('detect')
            plt.imshow(detectImage)
            '''
            plt.show()

if __name__ == '__main__':

    # Get the path of the trained models and test dataset
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--testingSet", help="Path to test dataset", required="True")
    parser.add_argument("-m", "--modelsTrained", help="Path to trained models", required="True")
    args = vars(parser.parse_args())

    # Get the training path to labeled dataset
    test_path = args["testingSet"]
    training_names = os.listdir(test_path)
    models_path = args["modelsTrained"]

    clf, reg = loadModels(models_path, ["AdaClassifier", "AdaRegressor"])

    #recover training metadata
    useSobel, pyr_hight, h, w, hogCellsPerBlock,  hogPixelPerCell, hogOrientation = recoverMetadata(models_path)
    shift = 5 #shift of sliding windows in test

    #process images
    runTest(clf, reg, test_path, training_names, useSobel, pyr_hight, h, w, hogPixelPerCell, hogCellsPerBlock, hogOrientation)
