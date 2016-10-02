import argparse as ap
import cv2
import numpy as np
import os

from random import shuffle

from sklearn import svm, ensemble
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
    
    # Get the training path to labeled dataset
    train_path = args["dataset"]
    training_names = os.listdir(train_path)

    #create models    
    SVMclf = svm.SVC(C = 1000, kernel='rbf', gamma = 0.001)
    SVMrgr = svm.SVR(C = 1000, kernel='rbf', gamma = 0.001)
    ForestClassifier = ensemble.RandomForestClassifier()
    ForestRegressor = ensemble.RandomForestRegressor()
    ADAclasifier = ensemble.AdaBoostClassifier()
    ADAregressor = ensemble.AdaBoostRegressor()
    
    
    #makeClean(training_names, train_path)
    # Extract HOG features and set labels 
    classNumber = 5
    hogCellsPerBlock = (3, 3)
    hogPixelPerCell = (6, 6)    
    hogOrientation = 18
    cnt = np.zeros(classNumber)
    featureList = []
    labelList = []
    percentageList = []
    
    #create save directory
    directory = train_path + "_trained_" + str(classNumber) + "_" + str(hogOrientation) + "_" + str(hogCellsPerBlock) + "_" + str(hogPixelPerCell)
    if not os.path.exists(directory):
            os.makedirs(directory)
    '''
    for name in training_names:
        classID = getClass(int(getPercentage(name)), classNumber)
        imagePath = os.path.join(train_path, name)
        im = cv2.imread(imagePath)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        features = hog(im, orientations=hogOrientation,  pixels_per_cell=hogPixelPerCell, cells_per_block=hogCellsPerBlock, visualise=False)
        featureList.append(features)
        labelList.append(classID)
        percentageList.append(int(getPercentage(name)))
        cnt[classID]+=1
    print "Number of elements per class:"
    joblib.dump((featureList, labelList, percentageList, cnt), os.path.join(directory, "features_labels.pkl"))
    print cnt   
    ''' 
    print "reading"
    featureList, labelList, percentageList, cnt = joblib.load(os.path.join(directory, "features_labels.pkl"))
    print "readed"
    #'''
    
    #process the number f elements per class
    #tmp =  np.column_stack((featureList, labelList))
    #print tmp.shape[:2]
    n = min(cnt)
    print cnt
    print n
    #np.random.shuffle(tmp)
    tmpFeatureList = featureList
    tmpLabelList = labelList
    featureList = []
    labelList = []
    cnt = np.zeros(classNumber)
    print "getting right classess"
    for i in range(len(tmpFeatureList)):
        if(cnt[tmpLabelList[i]]<n):
            featureList.append(tmpFeatureList[i])
            labelList.append(tmpLabelList[i])
            cnt[tmpLabelList[i]]+=1
    tmpFeatureList = []
    tmpLabelList = []
    
    #train and save models
    print "Model Training has began"
    SVMclf.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(SVMclf, os.path.join(directory, "SVMclassifier.pkl"))
    print "SVMclassifier has been saved"
    
    print "Model Training has began"
    SVMrgr.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(SVMrgr, os.path.join(directory, "SVMregressor.pkl"))
    print "SVMregressor has been saved"
    
    print "Model Training has began"
    ForestClassifier.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(ForestClassifier, os.path.join(directory, "ForestClassifier.pkl"))
    print "ForestClassifier has been saved"
    
    print "Model Training has began"
    ForestRegressor.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(ForestRegressor, os.path.join(directory, "ForestRegressor.pkl"))
    print "ForestRegressor has been saved"
    
    print "Model Training has began"
    ADAclasifier.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(ADAclasifier, os.path.join(directory, "ADAclasifier.pkl")) 
    print "ADAclasifier has been saved"
    
    print "Model Training has began"
    ADAregressor.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(ADAregressor, os.path.join(directory, "ADAregressor.pkl"))
    print "ADAregressor has been saved"
    

    
    '''
    SVMclassifier
    SVMregressor
    ForestClassifier
    ForestRegressor
    ADAclasifier
    ADAregressor
    '''
