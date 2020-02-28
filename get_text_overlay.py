# This is the problem for First technical round for the role of Computer Vision Engineer at Vectorly
# More details at https://www.linkedin.com/jobs/view/1629909785/
#
# Write a function which will segment and extract the text overlay "Bart & Homer's EXCELLENT Adventure" 
# Input image is at https://vectorly.io/demo/simpsons_frame0.png
# Output : Image with only the overlay visible and everything else white
# 
# Note that you don't need to extract the text, the output is an image with only 
# the overlay visible and everything else (background) white
#
# You can use the snipped below (in python) to get started if you like 
# Python is not required but is preferred. You are free to use any libraries or any language


#####################
#
# I implemented two methods one using adaptive thresholding and the other using KMeans clustering 
# for segmentation. By default it is set to method 1 which implements KMeans clustering.
# The saved labels and center output for KMeans is stored in pickle file to maintain consistency.
# Also the steps can be visualized by setting showSteps to True.
#
#####################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import path

# displaying images in notebook
from pylab import rcParams
rcParams['figure.figsize'] = 50,20

def kMeansClustering(input_image):

    grayImage = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    pixel_values = grayImage.reshape((-1, 3))
    
    k = 3
    if not path.exists('kMeans_labels.p'):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        pickle.dump(labels,open('kMeans_labels.p',"wb"))
        pickle.dump(centers,open('kMeans_centers.p', "wb"))
    else:
        labels=pickle.load(open('kMeans_labels.p','rb'))
        centers=pickle.load(open('kMeans_centers.p','rb'))

    labels = labels.flatten()
    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    for cluster in range(1,k):
        masked_image[labels == cluster] = [255, 255, 255]
    masked_image = masked_image.reshape(image.shape)

    return masked_image

def thresholdByPerimeter(image, contours, hierarchy, threshold):

    selected = []
    for i,nodes in enumerate(hierarchy[0]):
        if nodes[2]==-1:
            selected.append(nodes[2])

    for cntNum,contour in enumerate(contours):
        if not [0, 0] in contour[0,0]:
            if (threshold<cv2.arcLength(contour,True) and cntNum not in selected):
                image = cv2.fillPoly(image, pts =[contour], color=(0,0,0))

    return image



def getTextOverlay(input_image, method=1, showSteps=False):
     
    if method == 1:

        segmentedImg = kMeansClustering(input_image)
        output = cv2.morphologyEx(segmentedImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15)))

        if showSteps:
            f,ax = plt.subplots(1,2)
            ax[0].imshow(segmentedImg,cmap='gray')
            ax[0].set_title('segmented Image')

            ax[1].imshow(output,cmap='gray')
            ax[1].set_title('output')
            plt.show()
    else:

        blur = cv2.GaussianBlur(input_image,(3,3),0)

        th1 = cv2.adaptiveThreshold(cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,99,1)
        th2 = cv2.adaptiveThreshold(cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,55,2)

        closing = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(17,17)))
        opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))

        contours,hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgContours = cv2.drawContours(input_image, contours, -1, (0,255,0), 3)

        filtContours = thresholdByPerimeter((np.ones(input_image.shape, dtype=np.uint8)*255), contours, hierarchy, 190)

        output = filtContours[:,:,0]+opening[:,:]

        if showSteps:
            f,ax = plt.subplots(4,2)
            ax[0,0].imshow(cv2.cvtColor(blur,cv2.COLOR_BGR2RGB))
            ax[0,0].set_title('blur')

            ax[0,1].imshow(th1,cmap='gray')
            ax[0,1].set_title('threshold for opening')

            ax[1,0].imshow(th2,cmap='gray')
            ax[1,0].set_title('threshold for closing')

            ax[1,1].imshow(opening,cmap='gray')
            ax[1,1].set_title('opening')

            ax[2,0].imshow(closing,cmap='gray')
            ax[2,0].set_title('closing')

            ax[2,1].imshow(cv2.cvtColor(imgContours,cv2.COLOR_BGR2RGB))
            ax[2,1].set_title('contours detected after closing')

            ax[3,0].imshow(cv2.cvtColor(filtContours,cv2.COLOR_BGR2RGB))
            ax[3,0].set_title('filtered contours filled')

            ax[3,1].imshow(output,cmap='gray')
            ax[3,1].set_title('output from opening + filtered')
            plt.show()

	     	        
    return output

if __name__ == '__main__':
    image = cv2.imread('simpsons_frame0.png')
    output = getTextOverlay(image, method=0, showSteps=True)
    cv2.imwrite('simpons_text.png', output)
#####################

