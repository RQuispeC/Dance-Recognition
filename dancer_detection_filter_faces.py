import argparse as ap
import numpy as np
import cv2
import time
from operator import itemgetter
import os
import Queue

from sklearn import svm, ensemble
from skimage.feature import hog, local_binary_pattern, greycomatrix, greycoprops
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.externals import joblib
import pylab as plt
from sklearn.preprocessing import normalize
from skimage.filters import roberts, sobel, scharr, prewitt, gaussian
from skimage.color import rgb2hed


import pandas
from skimage import io


from skimage import color
from skimage.transform import hough_circle, hough_ellipse
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter, ellipse_perimeter

from skimage.filters.rank import entropy
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray


dx = (0, 1, 0, -1, 1, -1, 1, -1)
dy = (1, 0, -1, 0, -1, 1, 1, -1)
path_detection_rf = '/home/rodolfo/Pictures/dances-data/ds4/RandomForest/'
path_detection_svm = '/home/rodolfo/Pictures/dances-data/ds4/SVM/'
path_detection_ada = '/home/rodolfo/Pictures/dances-data/ds4/AdaBoost/'


def inside(i, j, dfsImg):
    return i>=0 and j>=0 and i < dfsImg.shape[0] and j < dfsImg.shape[1]

def bfs(i, j, maxI, maxJ, minI, minJ, threshold, visited, dfsImg):
    visited[i][j] = 1
    q = Queue.Queue(maxsize=0)
    q.put((i, j))
    while(not q.empty()):
        cur = q.get()
        visited[cur[0]][cur[1]] = 1
        maxI = max(maxI, cur[0])
        maxJ = max(maxJ, cur[1])
        minI = min(minI, cur[0])
        minJ = min(minJ, cur[1])
        for k in range(8):
            ni = cur[0] + dx[k]
            nj = cur[1] + dy[k]
            if(inside(ni, nj, dfsImg) and visited[ni][nj] == 0 and dfsImg[ni][nj] >= threshold):
                visited[ni][nj] = 1
                q.put((ni, nj))
    return [minJ, minI, maxJ, maxI], visited

def non_max_suppression_fast(boxes, overlapThresh):
	# Malisiewicz et al.
    # if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

def getRGBpath(rgb_path, image_name):
    i = image_name.find("bin")
    image_name= image_name[:i] + "rgb" + image_name[i+3:];
    return os.path.join(rgb_path, image_name)

def query(ac_sum, topLeft_i, topLeft_j, bottomRight_i, bottomRight_j):
    if(topLeft_i==0 and topLeft_j==0):
        return ac_sum[bottomRight_i][bottomRight_j]
    if(topLeft_i==0):
        return ac_sum[bottomRight_i][bottomRight_j] - ac_sum[bottomRight_i][topLeft_j - 1]
    if(topLeft_j==0):
        return ac_sum[bottomRight_i][bottomRight_j] - ac_sum[topLeft_i - 1][bottomRight_j]
    return ac_sum[bottomRight_i][bottomRight_j] - ac_sum[topLeft_i - 1][bottomRight_j] - ac_sum[bottomRight_i][topLeft_j - 1] + ac_sum[topLeft_i - 1][topLeft_j - 1]

def createAcumulative(bin_im):
    #create an acumulative matrix to make faster querys
    n, m = bin_im.shape[:2]
    ac_sum = np.zeros((n, m))
    bin_im = cv2.cvtColor(bin_im, cv2.COLOR_BGR2GRAY)

    for i in range(n):
        for j in range(m):
            if(bin_im[i][j]<20):
                bit = 1;
            else:
                bit = 0;
            if(i==0 and j==0):
                ac_sum[i][j] = bit;
            elif(i==0):
                ac_sum[i][j] = ac_sum[i][j-1] + bit;
            elif(j==0):
                ac_sum[i][j] = ac_sum[i-1][j] + bit;
            else:
                ac_sum[i][j] = ac_sum[i-1][j] + ac_sum[i][j-1] - ac_sum[i-1][j-1] + bit;
    return ac_sum

def preprocess(img, plot, detected_img = [], name = "", type = 1):
    if type == 1:
        img = cv2.bilateralFilter(img,9, 75, 75)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        _ , _ , img  = cv2.split(img) #get V chanel
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,16)
    else:
        img = cv2.bilateralFilter(img,9, 75, 75)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = sobel(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    edges = canny(img, 3, 10, 40)
    if plot:
        fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
        ax0.imshow(img, cmap=plt.cm.gray)
        ax0.set_title('Preprocess')
        ax0.axis('off')

        ax1.imshow(edges, cmap=plt.cm.gray)
        ax1.set_title('Edges')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
        ax2.set_title('Detected')
        ax2.axis('off')

        plt.tight_layout()
        #plt.show()
        fig.set_size_inches(28, 26)
        fig.savefig(os.path.join('/home/rodolfo/Pictures/', name + '_SCA' + '.jpg'), bbox_inches='tight')

    return edges

def detectCircles(img, useSobel = True, type = 1):
    if useSobel :
        img = preprocess(img, False, type = type)
    else:
        img = canny(img, 2, 10, 40)

    # Detect two radii
    hough_radii = np.arange(50, 70, 2)
    hough_res = hough_circle(img, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract one circle
        num_peaks = 1
        peaks = peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    return accums, centers, radii

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


def runEigenFaces_Fisherfaces(originalImage, rectangles, name, plot = True):
    
    modelPath = 'models/eigenFaces_filter_justFaces_full.ylm'
    eigenFacesModel = cv2.face.createEigenFaceRecognizer()
    eigenFacesModel.load(modelPath)

    modelPath = 'models/fisherFaces_filter_justFaces_full.ylm'
    fisherFacesModel = cv2.face.createFisherFaceRecognizer()
    fisherFacesModel.load(modelPath)

    it = 1
    w, h = 50, 50
    shift = 5
    rectanglesImage = originalImage.copy()
    rectangles_ans = []
    for rectangle in rectangles:
        img = originalImage[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n, m = img.shape

        acc = 0.0
        for i in xrange(0, n-h, shift):
            for j in xrange(0, m-w, shift):
                cropped = img[i:i+h, j:j+w].copy()
                cropped = cv2.resize(cropped, (40, 40)) #convert to a 8 multiple dimension to let face recoginizer work propertly https://goo.gl/LI5zL7

                label, confidence =  eigenFacesModel.predict(cropped)

                if confidence < 1000 or label == 0:
                    continue
                else:
                    acc +=label*confidence
                    break

                label, confidence =  fisherFacesModel.predict(cropped)
                if label == 0:
                    continue
                else:
                    acc +=label*confidence
                    break
            
            if acc != 0.0:
                break

        if plot:
            #plot image with scores
            fig = plt.figure(figsize=(30, 20))
            a=fig.add_subplot(len(rectangles),2,it)
            a.set_title(str(it))
            plt.imshow(detectedImage)

            cv2.putText(rectanglesImage, str(it), (rectangle[0], rectangle[3]-5), 1, 1, plotColor , 2, cv2.LINE_AA )

        if acc != 0.0:
            rectangles_ans.append(rectangle)

        it += 1

    return rectangles_ans

def isInside(point , rectangle):
    return point[0] >= rectangle[0] and point[0] <= rectangle[2] and point[1] >= rectangle[1] and point[1] <= rectangle[3]

def intersect(lineX, lineY):
    return lineX[0] <= lineY[0] and lineY[0] <= lineX[2] and lineY[1] <= lineX[1] and lineX[1] <= lineY[3]

def cascadeLBPfilter(originalImage, rectangles, name, plot = True):
    face_cascade = cv2.CascadeClassifier('models/cascade.xml')
    gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    cascade_faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    faces = []
    for face in cascade_faces: #convert rectangle format
        face = (face[0], face[1], face[0] + face[2], face[1] + face[3])
        faces.append(face)

    scores = []
    for rectangle in rectangles:
        score = 0
        for face in faces:
            #check if there is an intersection
            intersection = False
            if isInside((face[0], face[1]), rectangle):
                intersection = True
            if isInside((face[0], face[3]), rectangle):
                intersection = True
            if isInside((face[2], face[1]), rectangle):
                intersection = True
            if isInside((face[2], face[3]), rectangle):
                intersection = True
            if isInside((rectangle[0], rectangle[1]), face) or isInside((rectangle[0], rectangle[3]), face) or isInside((rectangle[2], rectangle[1]), face) or isInside((rectangle[2], rectangle[3]), face) :
                intersection = True
            if intersect((rectangle[0], rectangle[1], rectangle[2], rectangle[1]), (face[0], face[1], face[0], face[3])):
                intersection = True

            if intersection:
                score += 1
        scores.append(score)

    maxScore = max(scores)
    rectanglesAns = []
    for score, rectangle in zip(scores, rectangles):
        if score == maxScore or True :
            plotColor = (150, 255, 150)
            cv2.rectangle(originalImage, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), plotColor, 2)
            cv2.putText(originalImage, str(score), (rectangle[0], rectangle[3]-5), 1, 1, plotColor , 2, cv2.LINE_AA )
        if score != 0:
            rectanglesAns.append(rectangle)

    if plot:
        cv2.imwrite(os.path.join('/home/rodolfo/Pictures/', name + '_LBP_cascade.jpg'), originalImage)

    return rectanglesAns


def printRectangles(img, rectangles, name, save_path  = '/home/rodolfo/Pictures/', sufix = ''):
    it = 1
    maxi = 0

    for rectangle in rectangles:
        plotColor = (150, 255, 150)
        cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), plotColor, 2)
        cv2.putText(img, str(it), (rectangle[0], rectangle[3]-5), 1, 1, plotColor , 2, cv2.LINE_AA )
        it += 1

    fileSavePath = os.path.join(save_path, name + sufix + '.jpg')
    print 'Saving plot in: ', fileSavePath, ' ...'
    cv2.imwrite(fileSavePath, img)
    print 'done'


def filterByPosition(img, rectanglesHOG):
    down_lim = 0.60 * (img.shape[0])
    left_lim = 0.07 * img.shape[1]
    right_lim = 0.93 * img.shape[1]
    rectanglesAns = []
    for rectangle in rectanglesHOG:
        if rectangle[1] < down_lim and rectangle[0] > left_lim and rectangle[2] < right_lim:
            rectanglesAns.append(rectangle)
    return rectanglesAns

def multiFilter(originalImage, rectangles, name, plot = True, useTexture = True):
    #filter by eigenfaces_justfaces, fisherFaces_justFaces, LBP cascade and position
    if len(rectangles) > 0:
        rectangles = filterByPosition(originalImage.copy(), rectangles)

    print '\t running Eigenfaces and Fisherfaces filter ...', len(rectangles), 
    if len(rectangles) > 0: 
        rectangles = runEigenFaces_Fisherfaces(originalImage.copy(), rectangles, name, plot = False)
    print '\t done'

    if useTexture:
        print '\t running LBP cascade filter ...', 
        if len(rectangles) > 0:
            rectangles = cascadeLBPfilter(originalImage.copy(), rectangles, name, plot = False)
        print '\t done'

    

    if plot:
        it = 1
        for rectangle in rectangles:
            plotColor = (150, 255, 150)
            cv2.rectangle(originalImage, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), plotColor, 2)
            cv2.putText(originalImage, str(it), (rectangle[0], rectangle[3]-5), 1, 1, plotColor , 2, cv2.LINE_AA )
            it += 1
        cv2.imwrite(os.path.join('/home/rodolfo/Pictures/', name + '_multi.jpg'), originalImage)

    return rectangles

def improveFaces(img, rectanglesHOG):
    it = 1
    grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_or = img.copy()

    newRectangles = []
    for rectangle in rectanglesHOG:    # draw rectangles
        cropped = img[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]]

        #detect circles and plot them
        accums, centers, radii = detectCircles(cropped, True, 1)
        image = color.gray2rgb(img)
        if len(accums) == 0:
            it = it + 1
            continue
        if  max(accums)<0.05:
            it = it + 1
            continue
        limit = 5
        for idx in np.argsort(accums)[::-1][:limit]:
            center_x, center_y = centers[idx]
            center_x = center_x + rectangle[1]
            center_y = center_y + rectangle[0]
            radius = radii[idx]
            cx, cy = circle_perimeter(center_y, center_x, radius, shape = (image.shape[1], image.shape[0]))
            newRectangles.append((max(0, center_y - radius), max(0, center_x - radius), min(center_y + radius, img.shape[1]-1), min(center_x + radius, img.shape[0]-1)))

        it = it + 1

    newRectangles = np.array(newRectangles) #convert to nympy array
    overlapThresh = 0.5
    rectanglesHOG = non_max_suppression_fast(newRectangles, overlapThresh)

    return rectanglesHOG

def filterFalsePositives_noViolaJones(img, rectangles, name):
    #run multiFilter
    rectangles = multiFilter(img, rectangles, name, plot = False, useTexture = False)
    return rectangles

def filterFalsePositives(img, rectangles, name):
    #run multiFilter
    rectangles = multiFilter(img, rectangles, name, plot = False, useTexture = True)
    return rectangles

def pixelSize(rectangle):
    return (rectangle[2] - rectangle[0] + 1) * (rectangle[3] - rectangle[1] + 1)

def getLowBodyLimit(img):
    ans = img.shape[0] - 5
    last = img.shape[0]
    next = last - 10
    while next > 0:
        subImage = img[next:last, :]
        n_bins = 7
        histogram, _  = np.histogram(subImage.ravel(), normed=True, bins=n_bins, range=(0, n_bins))
        low_entropy = histogram[0] + histogram[1]
        hight_entropy =  histogram[6] + histogram[5] + histogram[4] + histogram[3]
        if low_entropy < 0.40 or hight_entropy >= 0.25:
            break
        ans = next
        next -= 20
    return last - ((last - ans)/2)

def extentToBody(img, rectangles, percentage):
    width = img.shape[1]
    height = img.shape[0]
    ans = []

    #get entropy of image
    img = cv2.bilateralFilter(img,15, 105, 155)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropyImage = entropy(img, disk(7))
    lowLimit = getLowBodyLimit(entropyImage)
    it = 1
    
    regCount = []

    while len(rectangles) > 0:
        last = len(rectangles) - 1
        rectangle = rectangles[last]
        falta = width * percentage - (rectangle[2] - rectangle[0])

        if rectangle[2] <= 0.5 * width:
            x = rectangle[0]
            y = max(0.5 * width - rectangle[2], 0.000001)
            xx = falta/(1 + x/y)
            yy = falta - xx
        elif rectangle[0] >= 0.5 * width:
            x = rectangle[0] - 0.5 * width
            y = max(width - rectangle[2], 0.000001)
            xx = falta/(1 + x/y)
            yy = falta - xx
        else:
            x = rectangle[0]
            y = width - rectangle[2]
            yy = falta/(1 + x/y)
            xx = falta - yy

        rectangle[0] = max(20, rectangle[0] - xx)
        rectangle[2] = min(width - 20, rectangle[2] + yy)
        rectangle[3] = lowLimit

        regCount.append(rectangle)
        if (0.0 + rectangle[3] - rectangle[1])/( 0.0 + rectangle[2] - rectangle[0]) > 1.0:
            ans.append(rectangle)

        rectangles[last] = rectangle
        rectangles = non_max_suppression_fast(np.array(rectangles), 0.99)

        if len(regCount) >= len(rectangles):
            break
        it += 1

    return ans

def compressBody(img, rectangles):
    if len(rectangles) == 0:
        return []
    it = 0
    ind = 0
    maxi = 0
    maxUp = img.shape[0]
    for rectangle in rectangles:
        area = (0.0 + rectangle[3] - rectangle[1])*( 0.0 + rectangle[2] - rectangle[0])
        maxUp = min(maxUp, rectangle[1])
        if area > maxi:
            maxi = area
            ind = it
        it += 1
    ans = rectangles[ind]
    ans[1] = maxUp
    return [ans]

def jointResults(img, name, detected_faces):
    detected_faces[0].extend(detected_faces[1])

    print 'nms', len(detected_faces[0])
    faces = non_max_suppression_fast(np.array(detected_faces[0]), 0.80)

    bodyPercentage = 0.75

    print 'extending body', len(detected_faces[0])
    body = extentToBody(img, faces, bodyPercentage)
    print 'getting dancer'
    body = compressBody(img, body)

    if len(body) == 0:
        print '============has no body============='
        body = [(0, 0, img.shape[1], img.shape[0])]

    return body

def runImageBFS(threshold, detectedClf, minPixelSize, n, m):
    rectanglesClf = []
    visited = np.zeros(detectedClf.shape[:2])
    for i in range(n):
        for j in range (m):
            if(visited[i][j]==0 and detectedClf[i][j] >= threshold):
                rectangle, visited = bfs(i, j, 0, 0, n+m, n+m, threshold, visited, detectedClf)
                if(pixelSize(rectangle) > minPixelSize):
                    rectanglesClf.append(rectangle)
    return rectanglesClf

def detectDancer(test_path, training_names, dataFiles_ada, dataFiles_rf, dataFiles_svm, plot = False):
    training_names.sort()
    dfs_threshold = [0.70, 0.50, 0.10]
    for name in training_names:
        extension = name[name.rfind('.'):]
        if extension == '.txt':
            continue

        img = cv2.imread(os.path.join(test_path, name))
        n, m = img.shape[:2]

        normal_detected_faces_ada = []
        normal_detected_faces_rf = []
        normal_detected_faces_svm = []

        noColor_noEdges_detected_faces_ada = []
        noColor_noEdges_detected_faces_rf = []
        noColor_noEdges_detected_faces_svm = []

        noTexture_detected_faces_ada  = []
        noTexture_detected_faces_rf  = []        
        noTexture_detected_faces_svm = []

        print 'Working on ', name
        if os.path.isfile(path_detection_svm + 'normal/'+ name[:name.rfind('.')] + '.txt'):
            print name, 'body has been already detected'
            continue

        for ind in range(1, 3):
            print 'Current pyramid level:   ', ind
            file_ada = np.load(os.path.join(dataFiles_ada, name[:name.rfind('.')]+'_' + str(ind) + '.npz'))
            file_rf = np.load(os.path.join(dataFiles_rf, name[:name.rfind('.')]+'_' + str(ind) + '.npz'))            
            file_svm = np.load(os.path.join(dataFiles_svm, name[:name.rfind('.')]+'_' + str(ind) + '.npz'))

            detectedClf_ada = cv2.resize(file_ada['arr_0'], (m, n))
            detectedClf_rf = cv2.resize(file_rf['arr_0'], (m, n))            
            detectedClf_svm = cv2.resize(file_svm['arr_0'], (m, n))

            '''
            fig = plt.figure(figsize=(30, 20))
            
            a=fig.add_subplot(1,3,1)
            a.set_title("adaboost")
            plt.imshow(detectedClf_ada)
            a.axis("off")

            a=fig.add_subplot(1,3,2)
            a.set_title("random forest")
            plt.imshow(detectedClf_rf)
            a.axis("off")

            a=fig.add_subplot(1,3,3)
            a.set_title("svm")
            plt.imshow(detectedClf_svm)
            a.axis("off")

            plt.show()
            '''

            #create rectangles based on previus results
            minPixelSize = 1000
            threshold = int((dfs_threshold[ind-1])*detectedClf_ada.max())
            visited = np.zeros(detectedClf_ada.shape[:2])
            rectanglesClf_ada = runImageBFS(threshold, detectedClf_ada, minPixelSize, n, m)

            minPixelSize = 1000
            threshold = int((dfs_threshold[ind-1])*detectedClf_rf.max())
            visited = np.zeros(detectedClf_rf.shape[:2])
            rectanglesClf_rf = runImageBFS(threshold, detectedClf_rf, minPixelSize, n, m)

            minPixelSize = 1000
            threshold = int((dfs_threshold[ind-1])*detectedClf_svm.max())
            visited = np.zeros(detectedClf_svm.shape[:2])
            rectanglesClf_svm = runImageBFS(threshold, detectedClf_svm, minPixelSize, n, m)

            faces_ada = improveFaces(img.copy(), rectanglesClf_ada) #color y bordes
            faces_rf = improveFaces(img.copy(), rectanglesClf_rf) #color y bordes
            faces_svm = improveFaces(img.copy(), rectanglesClf_svm) #color y bordes
            
            #normal detection
            normal_filtered_faces_ada = filterFalsePositives(img, faces_ada, name[:name.rfind('.')])
            normal_filtered_faces_rf = filterFalsePositives(img, faces_rf, name[:name.rfind('.')])
            normal_filtered_faces_svm = filterFalsePositives(img, faces_svm, name[:name.rfind('.')])

            normal_detected_faces_ada.append(normal_filtered_faces_ada)
            normal_detected_faces_rf.append(normal_filtered_faces_rf)
            normal_detected_faces_svm.append(normal_filtered_faces_svm)

            #  no color , no edges detection
            noColor_noEdges_filtered_faces_ada = filterFalsePositives(img, rectanglesClf_ada, name[:name.rfind('.')])
            noColor_noEdges_filtered_faces_rf = filterFalsePositives(img, rectanglesClf_rf, name[:name.rfind('.')])            
            noColor_noEdges_filtered_faces_svm = filterFalsePositives(img, rectanglesClf_svm, name[:name.rfind('.')])
            
            noColor_noEdges_detected_faces_ada.append(noColor_noEdges_filtered_faces_ada)
            noColor_noEdges_detected_faces_rf.append(noColor_noEdges_filtered_faces_rf)          
            noColor_noEdges_detected_faces_svm.append(noColor_noEdges_filtered_faces_svm)

            #no texture detection
            noTexture_filtered_faces_ada  = filterFalsePositives_noViolaJones(img, faces_ada, name[:name.rfind('.')])
            noTexture_filtered_faces_rf  = filterFalsePositives_noViolaJones(img, faces_rf, name[:name.rfind('.')])            
            noTexture_filtered_faces_svm = filterFalsePositives_noViolaJones(img, faces_svm, name[:name.rfind('.')])

            noTexture_detected_faces_ada.append(noTexture_filtered_faces_ada)
            noTexture_detected_faces_rf.append(noTexture_filtered_faces_rf)            
            noTexture_detected_faces_svm.append(noTexture_filtered_faces_svm)

        # =============== ADABOOST ==============================
        print 'ADABOOST'

        print 'Joining results noColor no Edges'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noColor_noEdges_detected_faces_ada)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_ada + 'noColorNoEdges/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_adaboost_noColorNoEdges')

        #//////////////

        print 'Joining results no texture'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noTexture_detected_faces_ada)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_ada + 'noTexture/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_adaboost_noTexture')

        #//////////////

        print 'Joining results normal'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], normal_detected_faces_ada)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_ada + 'normal/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_adaboost_normal')

        # =============== RANDOM FOREST ==============================

        print 'RANDOM FOREST'

        print 'Joining results no Color no edges'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noColor_noEdges_detected_faces_rf)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_rf + 'noColorNoEdges/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_randomForest_noColorNoEdges')

        #//////////////

        print 'Joining results no texture'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noTexture_detected_faces_rf)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_rf + 'noTexture/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_randomForest_noTexture')

        #//////////////

        print 'Joining results normal'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], normal_detected_faces_rf)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_rf + 'normal/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_randomForest_normal')

        # =============== SVM ==============================

        print 'SVM'

        print 'Joining results no color no edges'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noColor_noEdges_detected_faces_svm)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_svm + 'noColorNoEdges/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_svm_noColorNoEdges')

        #//////////////

        print 'Joining results no texture'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], noTexture_detected_faces_svm)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_svm + 'noTexture/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_svm_noTexture')

        #//////////////

        print 'Joining results normal'
        dancers = jointResults(img.copy(), name[:name.rfind('.')], normal_detected_faces_svm)

        #save body location
        print 'Saving dancer locations ...'
        dancerLocationName = path_detection_svm + 'normal/' + name[:name.rfind('.')] + '.txt'
        dancerLocationFile = open(dancerLocationName, 'w')
        for dancer in dancers:
             dancerLocationFile.write(str(dancer[0]) + ' ' + str(dancer[1]) + ' ' + str(dancer[2]) + ' ' + str(dancer[3]))
        print 'done'

        if plot:
            printRectangles(img.copy(), dancers, name[:name.rfind('.')] , sufix = '_body_svm_normal')

        

def createDirectories():
    directory = path_detection_ada + 'normal/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_ada + 'noColorNoEdges/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_ada + 'noTexture/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = path_detection_rf + 'normal/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_rf + 'noColorNoEdges/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_rf + 'noTexture/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    directory = path_detection_svm + 'normal/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_svm + 'noColorNoEdges/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = path_detection_svm + 'noTexture/'
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    createDirectories()

    # ====================== path to .npz files result of hog pyramid =================================    
    dataFiles_ada = '/home/rodolfo/Pictures/dances-data/ds4/results/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_AdaBoost_n_estimators_150/'
    dataFiles_rf  = '/home/rodolfo/Pictures/dances-data/ds4/results/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_RandomForest_n_estimators_150/'
    dataFiles_svm = '/home/rodolfo/Pictures/dances-data/ds4/results/3_50_50_20_noSharper_trained_2_9_(2, 2)_(9, 9)_30_SVM_C_100_gamma_0.001/'
    

    # ====================== path to images =================================
    # Get the path of the trained models and test dataset
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--testingSet", help="Path to test dataset", required="True")
    args = vars(parser.parse_args())
    test_path = args["testingSet"]

    training_names = os.listdir(test_path)

    detectDancer(test_path, training_names, dataFiles_ada, dataFiles_rf, dataFiles_svm,  plot = 'True')
