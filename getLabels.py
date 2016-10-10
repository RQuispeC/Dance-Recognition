import argparse as ap
import cv2
import numpy as np
import os
from skimage.filters import sobel

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

def createDefaultDirectory(train_path = "", w = 50, h = 50, pyr_hight = 3, shift = 20, useSobel = True):
    sob = ""
    if(useSobel):
        sob = "_sobel"
    else:
        sob = "_noSharper"
    #create a directory
    tmp_path =os.path.basename(os.path.normpath(train_path))
    patches_path = train_path[:train_path.find(tmp_path)]
    directory = os.path.join(patches_path, "patches/"+str(pyr_hight)+"_"+str(h)+"_"+str(w)+"_"+str(shift)+sob)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def createDirectory(saveDirectory = "", w = 50, h = 50, pyr_hight = 3, shift = 20, useSobel = True):
    sob = ""
    if(useSobel):
        sob = "_sobel"
    else:
        sob = "_noSharper"
    directory = os.path.join(saveDirectory,"patches/"+str(pyr_hight)+"_"+str(h)+"_"+str(w)+"_"+str(shift)+sob)
    print "Data will be saved in: ", directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def getLabelsByPercentage(train_path, training_names, directory,  bin_path, rgb_path, w = 50, h = 50, pyr_hight = 3, shift = 20, useSobel = True):
    #process images
    for name in training_names:
        #read images
        image_path = os.path.join(bin_path, name)
        bin_im_or = cv2.imread(image_path)
        if(useSobel):
            rgb_im_or = cv2.imread(getRGBpath(rgb_path, name), False)
            rgb_im_or = sobel(rgb_im_or)
            rgb_im_or = cv2.normalize(rgb_im_or, None, 0, 255, cv2.NORM_MINMAX)
        else:
            rgb_im_or = cv2.imread(getRGBpath(rgb_path, name))
            rgb_im_or = cv2.cvtColor(rgb_im_or, cv2.COLOR_BGR2GRAY)
        print name
        print bin_im_or.shape[:2]
        print rgb_im_or.shape[:2]

        #slide the window and create the archives with label
        it = 1
        cnt = np.zeros(101)
        for k in range(pyr_hight):
            #resize images and get acumulative matrix
            bin_im = cv2.resize(bin_im_or, (0,0), fx = 1.0/(k+1.0), fy = 1.0/(k+1.0));
            rgb_im = cv2.resize(rgb_im_or, (0,0), fx = 1.0/(k+1.0), fy = 1.0/(k+1.0));
            print "Factor : ", k+1
            n, m = bin_im.shape[:2]
            if(n<h or m<w):
                continue
            n, m = rgb_im.shape[:2]
            if(n<h or m<w):
                continue

            acumulative = createAcumulative(bin_im)
            tot = acumulative[n-1][m-1]
            for i in range(0, n-h, shift):
                for j in range(0, m-w, shift):
                    percentage = int((query(acumulative, i, j, i+h-1, j+w-1)*100.0)/(1.0*tot))
                    cropped = rgb_im[i:i+h, j:j+w]
                    pos = name.find("_")
                    pos = name.find("_", pos+1)
                    saveName = name[:pos+1]
                    saveName+=str(percentage)+"_"+str(it)+".jpg"
                    it+=1
                    cv2.imwrite(os.path.join(directory, saveName), cropped)
                    cnt[percentage]+=1
                    #print cropped.shape[:2]
        print cnt

if __name__ == '__main__':
    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to train dataset", required="True")
    parser.add_argument("-s", "--saveDirectory", help="Path where you want to save the labeled patches")
    args = vars(parser.parse_args())

    # Get the training path to binary and rgb images
    train_path = args["dataset"]
    bin_path = os.path.join(train_path, "bin")
    rgb_path = os.path.join(train_path, "rgb")
    training_names = os.listdir(bin_path)
    print "Path to binary images : ", bin_path
    print "Path to rgb images : ", rgb_path

    #define windows size, pyramid hight and windows shift
    w, h = 50, 50
    pyr_hight = 3
    shift = 20
    useSobel =True

    #create directory to save patches
    if(args["saveDirectory"]):
        directory = createDirectory(args["saveDirectory"], w, h, pyr_hight, shift, useSobel)
    else:
        directory = createDefaultDirectory(train_path, w, h, pyr_hight, shift, useSobel)

    #create label
    #getLabelsByPercentage(train_path, training_names, bin_path, rgb_path)
    getLabelsByPercentage(train_path, training_names, directory, bin_path, rgb_path, w, h, pyr_hight, shift, useSobel)
