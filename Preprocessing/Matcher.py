from skimage import io ,transform ,feature,measure,filters,morphology,transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import cv2
import glob
from commonfunctions import *
def match(img,templates,threshold):
        img_width, img_height = img.shape
        best_location_count = -1
        best_locations = []
        best_scale = 1

        for template in templates:
                locations = []
                template= rgb2gray(template)

                for scale in [i/100.0 for i in range(50, 150, 3)]:

                    
                    resizex = int(scale*template.shape[0])
                    resizey = int(scale*template.shape[1])
                    if ( resizex > img_width) or ( resizey > img_height):
                        continue

  
                    # Perform match operations.

                    
                    template = cv2.resize(template, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
                    template=template.astype(np.float32)
                    img=img.astype(np.float32)
                    res  = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        
                    loc = np.where(res >= threshold)
            
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(img, pt, (pt[0] + resizex, pt[1] + resizey), (0,0,255), 2)

                        cv2.imwrite('res.png',img)


  

        return best_locations, best_scale