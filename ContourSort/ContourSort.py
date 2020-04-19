# sorting the given contours

import numpy as np
import argparse
import imutils
import cv2

def sortContours(cnts, method="left-to-right"):
    reverse = False
    i=0

    # handle if sorting is goinf to be reverse
    if method=="right-to-left" or method=="bottom-to-top":
        reverse = True

    # handle if sorting is to be done in y direction
    if method=="top-to-bottom" or method=="bottom to top":
        i=1

    # constructing a list of bounding ectangles
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    print(boundingBoxes)



    return (cnts,boundingBoxes)

def drawContours(image,c,i):
    # draw the centre of contour
    M=cv2.moments(c)
    cX=int(M["m10"]/M["m00"])
    cY=int(M["m01"]/M["m00"])
    cv2.circle(image,(cX,cY),3,(255,0,0),-1)


    # draw the contour no.
    cv2.putText(image,"#{}".format(i+1),(cX-20,cY+10),cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,255,0),2)

    return image

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="input image path")
ap.add_argument("-m","--method",required=True,help="sorting method")

args = vars(ap.parse_args())
print(args)
image = cv2.imread(args["image"])
(h,w,d)=image.shape
image = cv2.resize(image,(int(w*.8),int(h*.8)))
accumEdged = np.zeros(image.shape[:2],dtype='uint8')
print(accumEdged.shape)

for chan in cv2.split(image):

    chan=cv2.medianBlur(chan,15)
    cv2.imshow("channel",chan)
    edged = cv2.Canny(chan,90,240)
    cv2.imshow("chan_edge",edged)
    accumEdged = cv2.bitwise_or(accumEdged, edged)
    cv2.imshow("accum",accumEdged)
    cv2.waitKey(0)

cv2.imshow("edged map",accumEdged)

# find contours and eliminate all small ones
cnts = cv2.findContours(accumEdged.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
allcnt=image.copy()
for c in cnts:
    cv2.drawContours(allcnt,[c],-1,(0,0,255),2)
    cv2.imshow("all cntrs",allcnt)
    cv2.waitKey(0)

cnts = sorted(cnts,key=cv2.contourArea,reverse=False)[:5]
orig=image.copy()
redcnt=image.copy()
for c in cnts:
    cv2.drawContours(redcnt,[c],-1,(0,0,255),2)
cv2.imshow("red cntrs",redcnt)

# drawing unsorted contours
for (i,c) in enumerate(cnts):
    orig = drawContours(orig,c,i)
cv2.imshow("unsorted",orig)

#sort the contours
(cnts, boundingBoxes)=sortContours(cnts,method=args["method"])

# draw sorted contours
for (i,c) in enumerate(cnts):
    orig = drawContours(image,c,i)
cv2.imshow("sorted",image)

cv2.waitKey(0)
