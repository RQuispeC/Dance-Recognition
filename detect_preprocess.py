import dancer_detection_filter_faces as face_detection

import numpy as np
import cv2
import os
from skimage.transform import hough_circle, hough_ellipse
from skimage.feature import peak_local_max, canny
import pylab as plt

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
import gc

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

def edges_without_preprocess(img, path, name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = canny(gray, 3, 10, 40)
    print path  + 'edge_'+name


    fig = plt.figure(figsize=(30, 20))
    a=fig.add_subplot(1,2,1)
    a.axis("off")
    a.set_title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    a=fig.add_subplot(1,2,2)
    a.axis("off")
    a.set_title("Bordes")
    plt.imshow(edges, cmap=plt.cm.gray)

    fig.savefig(path  + 'edge_'+name, bbox_inches='tight')

def edges_with_preprocess(img, path = "", name = ""):
    fig = plt.figure(figsize=(30, 20))

    a=fig.add_subplot(1,5,1)
    a.axis("off")
    a.set_title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = cv2.bilateralFilter(img,9, 75, 75)
    a=fig.add_subplot(1,5,2)
    a.axis("off")
    a.set_title("Bilateral")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _ , _ , img  = cv2.split(img) #get V chanel
    a=fig.add_subplot(1,5,3)
    a.axis("off")
    a.set_title("Canal V")
    plt.imshow(img, cmap=plt.cm.gray)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,16)
    a=fig.add_subplot(1,5,4)
    a.axis("off")
    a.set_title("Thresholding")
    plt.imshow(img, cmap=plt.cm.gray)

    edges = canny(img, 3, 10, 40)
    a=fig.add_subplot(1,5,5)
    a.axis("off")
    a.set_title("Bordes")
    plt.imshow(edges, cmap=plt.cm.gray)

    print path  + 'edge_with_'+name
    #fig.savefig(path  + 'edge_with_'+name, bbox_inches='tight')

     #free ram 
    fig.clf()
    plt.close()
    del a
    gc.collect()

    return edges

def detectCircles(img, useSobel = True, type = 1):
    img = edges_with_preprocess(img)

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

def printRectangles(img, rectangles):
    it = 1
    maxi = 0

    for rectangle in rectangles:
        plotColor = (150, 255, 150)
        cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), plotColor, 2)
        it += 1

    return img

def improveFaces(img, rectanglesHOG, image_name):
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
            image[cy, cx] = (220, 20, 20)
            newRectangles.append((max(0, center_y - radius), max(0, center_x - radius), min(center_y + radius, img.shape[1]-1), min(center_x + radius, img.shape[0]-1)))

        it = it + 1
    cv2.imwrite('/tmp/'+ image_name + '_circles.jpg', printRectangles(image, rectanglesHOG))

    newRectangles = np.array(newRectangles) #convert to nympy array
    cv2.imwrite('/tmp/'+ image_name + '_rectangles.jpg', printRectangles(img_or.copy(), newRectangles))
    overlapThresh = 0.5
    rectanglesHOG = non_max_suppression_fast(newRectangles, overlapThresh)
    cv2.imwrite('/tmp/'+ image_name + '_rectangles_nms.jpg', printRectangles(img_or.copy(), rectanglesHOG))

    return rectanglesHOG

def circles(img, hog_files_path, image_name):
    dfs_threshold = [0.70, 0.50, 0.10]
    n, m = img.shape[:2]
    detected_hog = []
    detected_rectangles = []

    for ind in range(1, 3):
        #recover hog results
        print 'Current pyramid level:   ', ind
        file = np.load(os.path.join(hog_files_path, image_name+'_' + str(ind) + '.npz'))
        detectedClf = cv2.resize(file['arr_0'], (m, n))
        detected_hog.append(detectedClf)

        minPixelSize = 1000
        threshold = int((dfs_threshold[ind-1])*detectedClf.max())
        visited = np.zeros(detectedClf.shape[:2])
        rectanglesClf = face_detection.runImageBFS(threshold, detectedClf, minPixelSize, n, m)
        detected_rectangles.append(rectanglesClf)

    improveFaces(img.copy(), detected_rectangles[0], image_name)

    return detected_rectangles    

def cascadeLBPfilter(originalImage, name):
    face_cascade = cv2.CascadeClassifier('models/cascade.xml')
    gray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    cascade_faces = face_cascade.detectMultiScale(gray, 1.3, 3)
    faces = []
    for face in cascade_faces: #convert rectangle format
        face = (face[0], face[1], face[0] + face[2], face[1] + face[3])
        faces.append(face)
    
    cv2.imwrite('/tmp/'+ name + '_viola.jpg', printRectangles(originalImage, faces))
    

if __name__ == '__main__':

    '''
    path = '/home/rodolfo/Pictures/ds2/test/'
    name = 'chuncho_003.JPG'
    img = cv2.imread(path  + name)

    #edges_without_preprocess(img, path, name)

    edges_with_preprocess(img, path, name)
    '''

    images_path = '/home/rodolfo/Pictures/demo-data/images/' 

    hog_files_path = '/home/rodolfo/Pictures/demo-data/hog-files/'

    image_names = os.listdir(images_path)
    image_names.sort()

    for image_name in image_names:
        img = cv2.imread(images_path + image_name)
        #rectangles = circles(img, hog_files_path, image_name[:image_name.rfind('.')])
        cascadeLBPfilter(img, image_name[:image_name.rfind('.')])
