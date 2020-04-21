import cv2
from shapeDetectorFn import ShapeDetector
import imutils

sd=ShapeDetector()

image=cv2.imread('image1.jpg')
cv2.imshow("image",image)

# obtaining threshold image.
gray = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh ,(3,3))
thresh = cv2.dilate(thresh ,(3,3))

cv2.imshow("thresh",thresh)
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

shape="l"

for c in cnts:
    #find centre
    M=cv2.moments(c)

    # if to avoid undesirable contours
    if M["m00"] == 0:
        pass
    else:
        cX=int(M["m10"]/M["m00"])
        cY=int(M["m01"]/M["m00"])
        shape = sd.detect(c)
    
        cv2.drawContours(image,[c],-1,(255,0,200),2)
        cv2.putText(image, shape, (cX,cY),cv2.FONT_HERSHEY_SIMPLEX,
                    .5,(255,0,100),2)

        cv2.imshow("image",image)
        cv2.waitKey(0)

cv2.waitKey(0)
