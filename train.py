import argparse as ap
import cv2
import numpy as np
import os

# ==================  caffe  ======================================
caffe_root = '/home/rodolfo/Downloads/image-software/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# ===============================================================

from random import shuffle

from sklearn import svm, ensemble
from skimage.feature import hog
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.externals import joblib
from skimage.filters import sobel

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

def getBinaryClass(percentage, threshold):
    if(percentage>=threshold):
        return 1
    return 0

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

def resizeToNetInputSize(im):
    x, y = im.shape[:2]
    if(x == 50 and y == 50):#set resize case for a 50x50 window
        tmp_im = im
        im = np.concatenate((im, im), axis = 0) #create a 100x50 matrix
        im = np.concatenate((im, im), axis = 0) #create a 200x50 matrix
        im = np.concatenate((im, tmp_im), axis = 0) #create a 250x50 matrix
        tmp_im = im
        im = np.concatenate((im, im), axis = 1) #create a 250x100 matrix
        im = np.concatenate((im, im), axis = 1) #create a 200x200 matrix
        im = np.concatenate((im, tmp_im), axis = 1) #create a 250x250 matrix
    return  cv2.resize(im, (231, 231))

def readFeatures(directory, classNumber = 5):
    #read features already calculated
    print "reading"
    featureList, labelList, percentageList, cnt = joblib.load(os.path.join(directory, "features_labels.pkl"))
    print "readed"

    #balance the number of elements per class when data features are readed
    n = min(cnt)
    print cnt
    print n
    tmpFeatureList = featureList
    tmpLabelList = labelList
    featureList = []
    labelList = []
    cnt = np.zeros(classNumber)
    print "getting right classes"
    for i in range(len(tmpFeatureList)):
        if(cnt[tmpLabelList[i]]<n):
            featureList.append(tmpFeatureList[i])
            labelList.append(tmpLabelList[i])
            cnt[tmpLabelList[i]]+=1
    tmpFeatureList = []
    tmpLabelList = []
    return featureList, labelList, percentageList, cnt

def createDefaultDirectory(train_path, classNumber=2, threshold = 40, extraData = "", networkName = "bvlc_reference_caffenet"):
    if(classNumber != 2):
        threshold = ""
    else:
        threshold = "_" + str(threshold)
    if(extraData != ""):
        extraData = "_" + extraData
    #create directory to save models
    tmp_path =os.path.basename(os.path.normpath(train_path))
    patches_matadata = tmp_path
    models_path = train_path[:train_path.find(tmp_path)]
    tmp_path =os.path.basename(os.path.normpath(models_path))
    models_path = train_path[:models_path.find(tmp_path)]

    directory = os.path.join(models_path, "models/" + patches_matadata + "_trained_" + str(classNumber) + "_" + networkName + threshold + extraData)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
            os.makedirs(directory)
    return directory

def createDirectory(saveDirectory, train_path, classNumber=2, threshold = 40, extraData = "", networkName = "bvlc_reference_caffenet"):
    if(classNumber != 2):
        threshold = ""
    else:
        threshold = "_" + str(threshold)
    if(extraData != ""):
        extraData = "_" + extraData
    #create directory to save models
    patches_matadata =os.path.basename(os.path.normpath(train_path))

    directory = os.path.join(saveDirectory, "models/" + patches_matadata + "_trained_" + str(classNumber) + "_" +  networkName + threshold + extraData)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
            os.makedirs(directory)
    return directory

def balanceClasses(training_names, classNumber = 2, threshold = 40):
    #balance the number of elements per class
    print "balancing class numbers"
    tmpTraining_names = training_names
    np.random.shuffle(tmpTraining_names)
    training_names = []
    cnt = np.zeros(classNumber)
    sortedData = []
    for name in tmpTraining_names:
        if(classNumber == 2):
            classID = getBinaryClass(int(getPercentage(name)), threshold)
        else:
            classID = getClass(int(getPercentage(name)), classNumber)
        sortedData.append((int(getPercentage(name)), name))
        cnt[classID]+=1
    print "initial class number"
    print cnt

    sortedData.sort()
    tmpTraining_names = []
    for it in range(len(sortedData)):
        tmpTraining_names.append(sortedData[it][1])

    lim = min(cnt)
    cnt = np.zeros(classNumber)
    for name in tmpTraining_names:
        if(classNumber == 2):
            classID = getBinaryClass(int(getPercentage(name)), threshold)
        else:
            classID = getClass(int(getPercentage(name)), classNumber)
        if(cnt[classID]<lim):
            training_names.append(name)
            cnt[classID]+=1
    tmpTraining_names = []
    return training_names

def loadModel(networkName = "bvlc_reference_caffenet"):
    if(networkName == "bvlc_reference_caffenet"):
        return '/home/rodolfo/Downloads/image-software/caffe-master/models/bvlc_reference_caffenet/deploy_conv3.prototxt'

def loadWeight(networkName = "bvlc_reference_caffenet"):
    if(networkName == "bvlc_reference_caffenet"):
        return '/home/rodolfo/Downloads/image-software/caffe-master/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

def extractFeatures(train_path, training_names, classNumber = 2, threshold = 40, networkName = "bvlc_reference_caffenet"):
    cnt = np.zeros(classNumber)
    featureList = []
    labelList = []
    percentageList = []
    print "extracting features"
    cnt = np.zeros(classNumber)

    #load imaginet network
    model = loadModel(networkName)
    weight = loadWeight(networkName)
    net = caffe.Net(model, caffe.TEST, weights = weight )
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/home/rodolfo/Downloads/image-software/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    for name in training_names:
        if(classNumber == 2):
            classID = getBinaryClass(int(getPercentage(name)), threshold)
        else:
            classID = getClass(int(getPercentage(name)), classNumber)
        imagePath = os.path.join(train_path, name)

        im = cv2.imread(imagePath)
        #resize image to expected input size
        im = resizeToNetInputSize(im)
        #load the image in the data layer
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        #compute and set feature dimension
        net.forward()
        features = np.mean(net.blobs['conv3'].data, axis = 0) #original output has (10, 384, 13, 13) shape, this is converted to (384, 13, 13)
        features = np.array(features).reshape((features.size))

        featureList.append(features)
        labelList.append(classID)
        percentageList.append(int(getPercentage(name)))
        cnt[classID]+=1
    print "Number of elements per class:"
    #joblib.dump((featureList, labelList, percentageList, cnt), os.path.join(directory, "features_labels.pkl"))
    print cnt
    return featureList, labelList

def trainModels(featureList, labelList, directory = "", model = 'SVM', params_model = dict(C=100, kernel='rbf', gamma=0.001)):
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

    clf = type_classifier[model](**params_model)
    print "Model training has began"
    clf.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(clf, os.path.join(directory, model + "Classifier.pkl"))
    print model + " classifier has been saved"

    reg = type_regressor[model](**params_model)
    print "Model training has began"
    reg.fit(featureList, labelList)
    print "saving model ..."
    joblib.dump(clf, os.path.join(directory, model + "Regressor.pkl"))
    print model + " regressor has been saved"

if __name__ == '__main__':
    #setup GPU mode for caffe
    caffe.set_mode_gpu()

    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to train dataset", required="True")
    parser.add_argument("-s", "--saveDirectory", help="Path where you want to save trained models and extracted features")
    args = vars(parser.parse_args())

    # Get the training path to labeled dataset
    train_path = args["dataset"]
    training_names = os.listdir(train_path)

    # set parameters
    classNumber = 2
    threshold = 80 # set this parameter if classNumber = 2
    binaryClass = True
    model = 'AdaBoost'
    params_model = dict(n_estimators = 150)
    networkName = "bvlc_reference_caffenet"

    #create directory to save models
    if(args["saveDirectory"]):
        directory = createDirectory(args["saveDirectory"], train_path, classNumber, threshold, model + '_' + '_'.join(np.array([[k, v] for k, v in zip(params_model.keys(), params_model.values())]).flatten()), networkName)
    else:
        directory = createDefaultDirectory(train_path, classNumber, threshold, model + '_' + '_'.join(np.array([[k, v] for k, v in zip(params_model.keys(), params_model.values())]).flatten()), networkName)

    #balance size of classes
    training_names = balanceClasses(training_names, classNumber, threshold)

    #get features
    featureList, labelList = extractFeatures(train_path, training_names, classNumber, threshold, networkName)

    #train and save models
    trainModels(featureList, labelList, directory, model, params_model)
