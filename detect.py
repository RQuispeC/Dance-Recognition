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
from skimage.filters import sobel

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
    w, h = 50, 50
    pyr_hight = 3
    shift = 20
    
    classNumber = 5
    hogCellsPerBlock = (2, 2)
    hogPixelPerCell = (9, 9)
    hogOrientation = 9
        
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
                    cropped = sobel(cropped)
                    cropped = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX)
                    features = hog(cropped, orientations=hogOrientation,  pixels_per_cell=hogPixelPerCell, cells_per_block=hogCellsPerBlock, visualise=False)
                    features = features.reshape(1, len(features)) #reshape to 2D array
                    prediction = SVMclf.predict(features)
                    print prediction
                    regression = SVMreg.predict(features)
                    print regression
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
            
            plt.show()
            cv2.waitKey(0)
                    
                    
