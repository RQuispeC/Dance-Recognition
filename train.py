import argparse as ap
import cv2
import numpy as np
import os

from skimage.feature import hog
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.externals import joblib

def getPercentage(fileName):
    fileName = fileName[fileName.find("_")+1:]
    fileName = fileName[fileName.find("_")+1:]
    fileName = fileName[:fileName.find("_")]
    return fileName
def getClass(percentage, classNumber):
    if(percentage==100):
        percentage-=1
    div = 100//classNumber
    return percentage//div
def makeClean(training_names, train_path):
    for name in training_names:
        imagePath = os.path.join(train_path, name)
        statinfo = os.stat(imagePath)
        if(statinfo.st_size == 0):
            print "Erased: ", name
            os.remove(imagePath)
        else:
            im = cv2.imread(imagePath)
            print im.shape[:2]
    
if __name__ == '__main__':
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to train dataset", required="True")
    args = vars(parser.parse_args())
    
    # Get the training path to binary and rgb images
    train_path = args["dataset"]
    training_names = os.listdir(train_path)


    #create save directory
    directory = train_path + "_trained_" + str(classNumber) + "_" + str(hogOrientation) + "_" + str(hogCellsPerBlock) + "_" + str(hogPixelPerCell)
    if not os.path.exists(directory):
            os.makedirs(directory)    

    #create models    
    SVMclassifier = SVC
    SVMregressor = SVR
    ForestClassifier = RandomForestClassifier
    ForestRegressor = RandomForestRegressor
    ADAclasifier = AdaBoostClassifier
    ADAregressor = AdaBoostRegressor
    
    
    #makeClean(training_names, train_path)
    # Extract HOG features and set labels 
    classNumber = 5
    hogCellsPerBlock = (3, 3)
    hogPixelPerCell = (6, 6)
    cnt = np.zeros(classNumber)
    hogOrientation = 18
    featureList = []
    labelList = []
    for name in training_names:
        classID = getClass(int(getPercentage(name)), classNumber)
        imagePath = os.path.join(train_path, name)
        im = cv2.imread(imagePath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        features = hog(im, orientations=hogOrientation,  pixels_per_cell=hogPixelPerCell, cells_per_block=hogCellsPerBlock, visualise=False)
        featureList.append(features)
        labelList.append(classID)
        cnt[classID]+=1
    print "Number of elements per class:"
    print cnt
    joblib.dump((featureList, labelList, cnt), os.path.join(directory, "features_labels.pkl"), compress=3)
    
    #train and save models
    SVMclassifier.fit(featureList, labelList)
    joblib.dump(SVMclassifier, os.path.join(directory, "SVMclassifier.pkl"), compress=3)
    
    SVMregressor.fit(featureList, labelList)
    joblib.dump(SVMregressor, os.path.join(directory, "SVMregressor.pkl"), compress=3)
    
    ForestClassifier.fit(featureList, labelList)
    joblib.dump(ForestClassifier, os.path.join(directory, "ForestClassifier.pkl"), compress=3)
    
    ForestRegressor.fit(featureList, labelList)
    joblib.dump(ForestRegressor, os.path.join(directory, "ForestRegressor.pkl"), compress=3)
    
    ADAclasifier.fit(featureList, labelList)
    joblib.dump(ADAclasifier, os.path.join(directory, "ADAclasifier.pkl"), compress=3) 
    
    ADAregressor.fit(featureList, labelList)
    joblib.dump(ADAregressor, os.path.join(directory, "ADAregressor.pkl"), compress=3)
    

    
    '''
    SVMclassifier
    SVMregressor
    ForestClassifier
    ForestRegressor
    ADAclasifier
    ADAregressor
    '''
