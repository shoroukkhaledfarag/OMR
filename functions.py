from skimage import io ,transform ,feature,measure,filters,exposure
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
import numpy as np

import glob
import cv2 as cv
from matplotlib import cm
from skimage.morphology import binary_erosion, binary_dilation,binary_opening, binary_closing,skeletonize, thin,area_closing,disk

import sys
np.set_printoptions(threshold=sys.maxsize)
from collections import Counter 
from skimage.transform import resize
from skimage.morphology import binary_erosion, binary_dilation, binary_closing,skeletonize, thin
from skimage.util import img_as_ubyte
from skimage.util.shape import view_as_windows

from skimage import io ,transform ,feature,measure,filters,exposure
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt


#--------------------------------------------------------------------------------------------------------------------------#

def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

#--------------------------------------------------------------------------------------------------------------------------#
def showHist(img):
    plt.figure()
    imgHist = histogram(img, nbins=256)
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')
#-------------------------------------------------------BINARIZATION-------------------------------------------------------#

def to_binary(img,val):
    threshold = np.copy(img)
    threshold[threshold<val]=0
    threshold[threshold>=val]=1
    return threshold

#--------------------------------------------------------------------------------------------------------------------------#

def binarize(img):
    threshold = filters.threshold_otsu(img)
    img = to_binary(img,threshold)
    return img

#--------------------------------------------------------------------------------------------------------------------------#

#-----------------------------------------STAFF DETECTION AND REMOVAL-------------------------------------------------------#

def Run_Length_Encoding(array):
    ones = 0 
    zeros = 0
    output =[]
    output_BW =[]
    for i in range(len(array)):
        
        if array[i] == 0:
            zeros+=1
        elif array[i] == 1:
            ones+=1

        if i+1 < len(array) :
            if array[i+1] != array[i]:
                if array[i] == 0:
                    output.append(zeros)
                    output_BW.append(0)
                    zeros=0
                elif array[i] == 1:
                    output.append(ones)
                    output_BW.append(1)
                    ones=0
                    
        if i+1 == len(array):
            if array[i] == 0:
                output.append(zeros)
                output_BW.append(0)
                zeros=0
            elif array[i] == 1:
                output.append(ones)
                output_BW.append(1)
                ones=0           
                
    return output,output_BW

#--------------------------------------------------------------------------------------------------------------------------#
def most_frequent(lst):
    x= Counter(lst).most_common(1)
    zz = [list(elem) for elem in x]
    for i in range(len(zz)):
        if i == 0:
            if zz[i][1] == 1:
                return min(lst)
        return zz[i][0]
    
#--------------------------------------------------------------------------------------------------------------------------#    
def calculate_reference_lengths(img):
    
    staffspaceheight_arr = []
    stafflineheight_arr = []
    for i in range(img.shape[1]):
        col=img[:,i]
        output,output_BW = Run_Length_Encoding(col)
        ones= []
        zeros = []
        for i in range(len(output)):
            if output_BW[i] ==1:
                ones.append(output[i])
            else:
                zeros.append(output[i])
        staffspaceheight_arr.append(most_frequent(ones))
        stafflineheight_arr.append(most_frequent(zeros))
        
    staffspaceheight = most_frequent(staffspaceheight_arr)
    stafflineheight = most_frequent(stafflineheight_arr )

    return  staffspaceheight , stafflineheight  


#--------------------------------------------------------------------------------------------------------------------------#

def segmentation(img):
    
    #apply closing 
    closedImage=binary_closing(1-img)
    
    #label the image
    label_image = measure.label(closedImage,connectivity=2,background=0)
    
    # show image after segmentation
#     show_images([img,label_image],['img','label_image'])
    
    segmented_notes=[] #list of the segmented notes
    titles=[] #list of strings for image show titles
    i=1 #number of segment for image show titles

    Dict = {} 
    #loop on each labeled segment
    for region in measure.regionprops(label_image):
        
        # take regions with large enough areas
        if region.area >= 50:
            
            # get rectangle around segmented notes
            minr, minc, maxr, maxc = region.bbox
            
            #take this part from original binarized image
            #note.append(img[minr:maxr+2,minc:maxc+2])
            Dict[minc]=[]
            Dict[minc].append(img[minr:maxr+2,minc:maxc+2])
            Dict[minc].append(minr)
            Dict[minc].append(maxr)
            
            #add it to the list
           
            titles.append(str(i))
            i+=1
    min_r=[]
    max_r=[]
    for i in sorted (Dict.keys()) : 
        segmented_notes.append(Dict[i][0])
        min_r.append(Dict[i][1])
        max_r.append(Dict[i][2])
         
    return segmented_notes,min_r,max_r


#--------------------------------------------------------------------------------------------------------------------------#

def deskew(img):
    is_rotated=False
    edges2 = feature.canny(img, sigma=0.6)
    hspace, angles, distances=transform.hough_line(edges2)
    accum, angles_2, dists = transform.hough_line_peaks(hspace, angles, distances)
    skewAngle = np.rad2deg(np.median(angles_2))
    rotatedImg = np.copy(img)
    if(abs(skewAngle)<90):
        rotatedImg = img_as_ubyte(transform.rotate(img, 90+skewAngle,clip = True,resize=True,cval= 1))
        is_rotated=True
    return rotatedImg,is_rotated

#--------------------------------------------------------------------------------------------------------------------------#

# Functions for staff line removal :

def get_staffline_rows(img,rows,cols):
    
    histogram=np.zeros(rows)
    histogram=np.sum(img==0,axis=1)
    is_staff=histogram> 0.6*cols
    staff_lines_indices=np.argwhere(is_staff)
    return staff_lines_indices


def remove_stafflines(img,staff_lines_indices):

    new=img
#     new[staff_lines_indices-1,:]=1
    new[staff_lines_indices,:]=1
#     new[staff_lines_indices+1,:]=1
    return new
#--------------------------------------------------------------------------------------------------------------------------#

def staff_removal_1(img):
    #         print('-------------------------gaussianbinarized')
    staffspaceheight , stafflineheight  = calculate_reference_lengths(img.astype('int'))
    #     print(stafflineheight, staffspaceheight)

    indices=get_staffline_rows(img,img.shape[0],img.shape[1]) #background abyad

    start = int(indices[0])
    sub_img = []
    for ind in indices:

        if int(ind) >= start:

            start = int(ind) - (2*staffspaceheight)
            end = int(ind)  + (5 - 1) * (stafflineheight + staffspaceheight) + 1+ (2*staffspaceheight)
            if start <0 :
                start = 0
            image_np = np.copy(img[start:end ,:])
            sub_img.append(image_np)
            start = end
    
    return sub_img,staffspaceheight,stafflineheight

#--------------------------------------------------------------------------------------------------------------------------#

def segmentation_1(sub_img,stafflineheight):

    stacks_segmented = []
    stacks_min_r = []
    stacks_max_r = []
    stacks_indices = []
  
    for i in range(len(sub_img)):
        #show_images([sub_img[i]])
        indices=get_staffline_rows(sub_img[i],sub_img[i].shape[0],sub_img[i].shape[1])
        #print(indices[0][0])

        Without_Staff_Lines=remove_stafflines(sub_img[i],indices) 

        Without_Staff_Lines=1-Without_Staff_Lines

        SE=np.ones((stafflineheight+2,1))
        Without_Staff_Lines=binary_closing(Without_Staff_Lines,SE)
        #         show_images([1-Without_Staff_Lines],["after filling gaps"])

        Without_Staff_Lines = filters.median(1-Without_Staff_Lines)
        #         show_images([Without_Staff_Lines],[ 'Median Filter'])

        
        
        segmented_notes,min_r,max_r=segmentation(Without_Staff_Lines)
        #print(max_r)
        stacks_segmented.append(segmented_notes)
        stacks_min_r.append(min_r)
        stacks_max_r.append(max_r)
        stacks_indices.append(indices)
        
    return stacks_segmented , stacks_min_r , stacks_max_r , stacks_indices



def segmentation_2(sub_img,stafflineheight):

    stacks_segmented = []
    stacks_min_r = []
    stacks_max_r = []
    stacks_indices = []
   
    for i in range(len(sub_img)):
        img=sub_img[i]
            
        white_pixels = np.array(np.where(img == 0))
        indices=white_pixels[:,0]

       
        SE=np.ones((7,1))
        img=binary_erosion(1-img,SE)

        img = filters.median(1-img)

        SE=np.ones((3,3))
        img=binary_dilation(img,SE)

        SE=np.ones((2,2))
        img=binary_closing(1-img,SE)

        SE=np.ones((9,1))
        img=binary_dilation(binary_dilation(img,SE),SE)

        SE=disk(4)
        img=binary_dilation(img,SE)

        segmented_notes,min_r,max_r=segmentation(1-img)
        stacks_min_r.append(min_r)
        stacks_max_r.append(max_r)
        stacks_indices.append(indices)     

        
    
        stacks_accepted = []
        for note in segmented_notes:
            SE=disk(2)
            img=binary_dilation(1-note,SE)
            img=1-img
            if(not(np.all(note==0))):
                stacks_accepted.append(img)
                
        stacks_indices.append(indices)
        stacks_segmented.append(stacks_accepted)     
    return stacks_segmented , stacks_min_r , stacks_max_r , stacks_indices

#--------------------------------------------------------------------------------------------------------------------------#


def feng_threshold(img, w_size1=15, w_size2=30,
                   k1=0.15, k2=0.01, alpha1=0.1):
   
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size1 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Obtaining windows
    padded_img = np.ones((rows + w_size1 - 1, cols + w_size1 - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size1, w_size1))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))

    # Obtaining local coordinates for std range calculations
    hw_size = w_size2 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 2) * (x2 - x1 + 2)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means2
    means2 = sums / l_size

    # Computing standard deviation range
    std_ranges = np.sqrt(sqr_sums / l_size - np.square(means2))

    # Computing normalized standard deviations and extra alpha parameters
    n_stds = stds / std_ranges
    n_sqr_std = np.square(n_stds)
    alpha2 = k1 * n_sqr_std
    alpha3 = k2 * n_sqr_std

    thresholds = ((1 - alpha1) * means + alpha2 * n_stds
                  * (means - mins) + alpha3 * mins)

    return thresholds




#--------------------------------------------------------------------------------------------------------------------------#



def get_staff_row_indicies_captured(img, stafflineheight, staffspaceheight):

    row, col = img.shape
    
    tolerance =17
    
    sumOfRows = np.sum(img, axis = 1)
    rowIndices = np.where(sumOfRows< (col-tolerance)*1)
    up = np.min(rowIndices)
    down = np.max(rowIndices)
    

    
    sumOfColoumns = np.sum(img, axis = 0)
    colIndices = np.where(sumOfColoumns< (row-tolerance)*1)
    left = np.min(colIndices)
    right = np.max(colIndices)


    afterCrop = img[up:down + 1, left:right + 1]
    afterCrop = np.asarray(afterCrop)
    plt.imshow(afterCrop, cmap='gray')
    cv.imwrite('output.png', afterCrop)


    rowIndicesShifted = np.roll(rowIndices, -1)
    rowIndicesShifted = rowIndicesShifted[0]

    transitionIndices = np.where(np.abs(rowIndices - rowIndicesShifted) > 10)
    transitionIndices = transitionIndices[1]

    rowIndices = rowIndices[0]

    downIndices= rowIndices[transitionIndices]

    transitionIndicesUp = np.insert(transitionIndices,0,-1)
    transitionIndicesUp = np.delete(transitionIndicesUp,-1)

    upIndices= rowIndices[transitionIndicesUp+1]

    img1 = []
    for i in range(transitionIndices.shape[0]):
        
        image_val = img[upIndices[i]-2:downIndices[i] + 3, left:right + 1]
        area = image_val.shape[0]*image_val.shape[1]
        if staffspaceheight> 100:
            staffspaceheight = 50
        if (((5 * (staffspaceheight+stafflineheight) )-staffspaceheight)*image_val.shape[1]) <= area:
            img1.append(image_val)

    
    

    return img1


#--------------------------------------------------------------------------------------------------------------------------#


def staff_removal_2(img):
    
    staffspaceheight , stafflineheight  = calculate_reference_lengths(img.astype('int'))
    sub_img = get_staff_row_indicies_captured(img, stafflineheight, staffspaceheight)
    
    return sub_img,staffspaceheight+8,stafflineheight+5


#-------------------------> classification starts here 
def countBlack (image):
    blacks = 0
    for color in image.flatten():
        if color == 0:
            blacks += 1
    return blacks

def zooning(img,div):
    img_arr=[]
    width=img.shape[0]
    height=img.shape[1]
    
    for i in range(div):
        for j in range(div):   
            img_arr.append(img[i*width//div : (i+1)*width//div ,j*height//div : (j+1)*height//div ])

    return img_arr

def getVolume(img):
    nrows=img.shape[0]
    ncols=img.shape[1]
    
    area=nrows*ncols
    black_area=countBlack(img)
    volume=black_area/area
    
    return volume

def calculateDistance(x1, x2):
    
    distance =np.linalg.norm(x1-x2)
    return distance

def KNN(test_point, training_features, labels, k):
   
    distarr=[]
    for i in range (training_features.shape[0]):
        f=calculateDistance(training_features[i,:],test_point)
        distarr.append(f)
        
        
    sortedarr=np.sort(distarr)
    
    classes=np.zeros(28)
    
    for i in range(k):
        result = np.where(distarr == sortedarr[i])
        for j in range(26):
            if(labels[result][0]==j+1):
                classes[j]+=1
            
            
    classification=np.argmax(classes)

  
    return classification+1

def read_data(file_name):
 
    data = np.genfromtxt(file_name, delimiter=',')
    return data

def get_feature_vector(img):
    feature_vector=[]
    nrows=img.shape[0]
    ncols=img.shape[1]
    aspect_ratio=ncols/nrows
    feature_vector.append(aspect_ratio)
    mom=cv.HuMoments(cv.moments((1-img).astype(np.uint8))).flatten()
    for m in mom:
        feature_vector.append(m)

    img_array=zooning(img,4)
    for i in range(len(img_array)):
        feature_vector.append(getVolume(img_array[i]))
    return feature_vector

