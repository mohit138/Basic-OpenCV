# color labeler fn.

# importing essential packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class ColorLabeler : 
    def __init__(self):
        # initialize the color dictionary, conytaining the color name
        # as the key and rgb tuple as the value.
        colors=OrderedDict({
            "red" : (255,0,0),
            "green" : (0,255,0),
            "blue" : (0,0,255),
            "yellow" : (255,255,0),
            "orange" : (255,165,0)})
        
        #allocate memory for L*a*b image and initialize color names list
        self.lab = np. zeros((len(colors),1,3), dtype = "uint8")
        self.colorNames = []
        
        for (i, (name,rgb)) in enumerate(colors.items()):
            # update the L*a*b array and color names list
            self.lab[i] = rgb
            self.colorNames.append(name)
            
        # convert the L*a*b array and the color names lsit to L*a*b
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)
        
    def label(self, image, c):
        #construct a maxsk for conyour, them compute the avg of L*a*b value 
        #for the masked region
        
        #We seperate out the the shape by applying mask over image, and then find mean of rgb values of lab image.
        mask = np.zeros(image.shape[:2],dtype = "uint8")
        cv2.imshow("1",mask)
        cv2.drawContours(mask , [c],-1,255,-1)
        cv2.imshow("2",mask)
        mask = cv2.erode(mask, None, iterations=2)
        cv2.imshow("3",mask)
        mean= cv2.mean(image, mask=mask)[:3]
        #print(mean)
        
        #print(self.lab)
        
        
        # initialize the min dist found thus far
        minDist = (np.inf , None)
        
        
        # Now in loop, we check the closeness of mean values with our pre defined color values, and the min 
        # distance will give us the closest approx color. 
        # eg. mean = (180,160,175)
        # dist with each color = 70,130,192,96,33
        # as dist is least near orange(ie. 33) ,
        # therefore the color is orange !
        
        
        # loop over known L*a*b color values 
        for (i,row) in enumerate(self.lab):
            # compute dist bet, L*a*b color values 
            # and mean of image.
            d = dist.euclidean(row[0],mean)
            #print(d)
            #print(i,row)
            if d<minDist[0]: 
                minDist=(d,i)
            
        return self.colorNames[minDist[1]]
            
        
        
        
            