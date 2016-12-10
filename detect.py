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

def loadModels(models_path, model='SVC'):
    type_classifier = dict(
        SVM = svm.SVC,
        RandomForest = ensemble.RandomForestClassifier,
        AdaBoost = ensemble.AdaBoostClassifier,
    )

    type_regressor = dict(
        SVM = svm.SVR,
        RandomForest = ensemble.RandomForestRegressor,
        AdaBoost = ensemble.AdaBoostRegressor,
    )

    if not type_classifier.has_key(model):
        print "Invalid classifier. Please provide one of these:"
        print type_classifier.keys()

    if not type_regressor.has_key(model):
        print "Invalid regressor. Please provide one of these:"
        print type_regressor.keys()

    print "Loading " + model + " classsifier ..."
    clf = joblib.load(os.path.join(models_path, model + "Classifier.pkl"))
    print model + " classifier loaded successfully"

    print "Loading " + model + " regressor ..."
    reg = joblib.load(os.path.join(models_path, model + "Regressor.pkl"))
    print model + " regressor loaded successfully"

    return clf, reg

def recoverMetadata(models_path):
    models_path =  os.path.basename(os.path.normpath(models_path))
    myMetadata = models_path.split('_')
    return myMetadata[4] == "sobel", int(myMetadata[0]), int(myMetadata[1]), int(myMetadata[2]), tuple(map(int, regex.findall(r'\d+', myMetadata[8]))), tuple(map(int, regex.findall(r'\d+', myMetadata[9]))), int(myMetadata[7])

def createDefaultDirectory(models_path):
    dirName =os.path.basename(os.path.normpath(models_path))
    models_path = models_path[:models_path.find(dirName)]
    tmp_path =os.path.basename(os.path.normpath(models_path))
    models_path = models_path[:models_path.find(tmp_path)]

    directory = os.path.join(models_path, "results/" + dirName)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
            os.makedirs(directory)
    return directory

def createDirectory(saveDirectory, models_path):
    dirName =os.path.basename(os.path.normpath(models_path))
    directory = os.path.join(saveDirectory, "results/" + dirName)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
            os.makedirs(directory)
    return directory

def runTest(clf, reg, test_path, training_names, saveDirectory = "", useSobel = True, pyr_hight = 3, h = 50, w = 50, params_hog = dict(orientations = 9, pixels_per_cell = (9, 9), cells_per_block = (2, 2))):
    print "Parameters are: "
    print useSobel, pyr_hight, h, w, params_hog.values()
    pyr_hight = 2

    for name in training_names:
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
            im = cv2.resize(im_or, (0,0), fx = 1.0/(k+1.0), fy = 1.0/(k+1.0))
            print "Factor : ", k+1
            n, m = im.shape[:2]
            if(n<h or m<w):
                continue
            detectImageclf = np.zeros((n, m))
            detectImagereg = np.zeros((n, m))
            for i in range(0, n-h, shift):
                for j in range(0, m-w, shift):
                    cropped = im[i:i+h, j:j+w]
                    features = hog(cropped, **params_hog)
                    features = features.reshape(1, len(features)) #reshape to 2D array
                    prediction = clf.predict(features)
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

            #plt.show()
            fig.savefig(os.path.join(saveDirectory, name[:name.rfind(".")] + "_" + str(k+1) + ".jpg"), bbox_inches='tight')


            np.savez(os.path.join(saveDirectory, name[:name.rfind(".")] + "_" + str(k+1) + ".npz"), detectImageclf, detectImagereg)

if __name__ == '__main__':

    # Get the path of the trained models and test dataset
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--testingSet", help="Path to test dataset", required="True")
    parser.add_argument("-m", "--modelsTrained", help="Path to trained models", required="True")
    parser.add_argument("-s", "--saveDirectory", help="Path where you want to save the results of the ")
    args = vars(parser.parse_args())

    # Get the training path to labeled dataset
    test_path = args["testingSet"]
    training_names = os.listdir(test_path)
    models_path = args["modelsTrained"]
    model_name = 'AdaBoost'

    clf, reg = loadModels(models_path, model_name)

    #recover training metadata
    useSobel, pyr_hight, h, w, hogCellsPerBlock,  hogPixelPerCell, hogOrientation = recoverMetadata(models_path)
    shift = 5 #shift of sliding windows in test

    #create directory file for results
    if(args["saveDirectory"]):
        directory = createDirectory(args["saveDirectory"], models_path)
    else:
        directory = createDefaultDirectory(models_path)

    #process images
    runTest(clf, reg, test_path, training_names, directory, useSobel, pyr_hight, h, w, dict(orientations = hogOrientation, pixels_per_cell = hogPixelPerCell, cells_per_block = hogCellsPerBlock))
