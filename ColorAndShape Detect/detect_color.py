# detect color and shape of objects !

from Func.shapeDetectorFn import ShapeDetector
from Func.colorlabeler import ColorLabeler

import imutils
import cv2

image =cv2.imread("image1.jpg")
resized = imutils.resize(image,width=500)
ratio = image.shape[0] / float(resized.shape[0])


blurred = cv2.GaussianBlur(resized, (5,5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred,cv2.COLOR_BGR2LAB)
cv2.imshow("lab", lab)
thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("thresh",thresh)
cv2.waitKey(0)


cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# initialize shape detector and color detector functions
sd =ShapeDetector()
cl =ColorLabeler()

# loop over every contour !
for c in cnts:
    M = cv2.moments(c)
    cX = int((M["m10"])/(M["m00"]))
    cY = int((M["m01"])/(M["m00"]))
    
    shape = sd.detect(c)
    color = cl.label(lab, c)
    
    text = "{}  {}".format(shape,color)
    cv2.drawContours(resized, [c], -1,(0,255,0),2)
    cv2.putText(resized,text,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2)
    
    cv2.imshow("Image",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()