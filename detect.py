import argparse as ap
import cv2
import numpy as np
import os

from sklearn import svm, ensemble
from skimage.feature import hog
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.externals import joblib
import pylab as plt

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
    
    #read trained models
    print "Loading SVM classsifier"
    SVMclf = joblib.load(os.path.join(models_path, "SVMclassifier.pkl"))
    print "SVM classsifier loaded"
    print "Loading SVM regressor"
    SVMreg = joblib.load(os.path.join(models_path, "SVMregressor.pkl"))
    print "SVM regressor loaded"
    
    #recover training metadata
    w, h = 100, 100
    pyr_hight = 3
    shift = 20
    
    classNumber = 5
    hogCellsPerBlock = (3, 3)
    hogPixelPerCell = (6, 6)
    hogOrientation = 18
        
    #process images
    for name in training_names:
        image_path = os.path.join(test_path, name)
        im_or = cv2.imread(image_path)
        im_or = cv2.cvtColor(im_or, cv2.COLOR_BGR2GRAY)
        for k in range(pyr_hight):
            im = cv2.resize(im_or, (0,0), fx = 1.0/(k+1), fy = 1/(k+1.0));
            print name
            print "Factor : ", k+1
            n, m = im.shape[:2]
            if(n<h or m<w):
                continue
            n, m = im.shape[:2]
            if(n<h or m<w):
                continue
            detectImageclf = np.zeros((n, m))
            detectImagereg = np.zeros((n, m))
            for i in range(0, n-h, shift):
                for j in range(0, m-w, shift):
                    cropped = im[i:i+h, j:j+w]
                    features = hog(cropped, orientations=hogOrientation,  pixels_per_cell=hogPixelPerCell, cells_per_block=hogCellsPerBlock, visualise=False)
                    features = features.reshape(1, len(features)) #reshape to 2D array
                    prediction = SVMclf.predict(features)
                    print prediction
                    regression = SVMreg.predict(features)
                    print regression
                    detectImageclf[i:i+h, j:j+w]+=prediction
                    detectImagereg[i:i+h, j:j+w]+=regression
            cv2.imshow("Original Image", im)
            plt.imshow(detectImageclf)
            plt.imshow(detectImagereg)
            plt.show()
            cv2.waitKey(0)
                    
                    
