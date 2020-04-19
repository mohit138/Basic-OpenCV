# detect extreme points of a contour

import imutils
import cv2

# converting to gray scale and applying blur to remove noise
image = cv2.imread("extreme_points_input.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)

# apply threshold !
thresh = cv2.threshold(gray,40,255,cv2.THRESH_BINARY)[1]

#erode and dialate to remove small noise
thresh=cv2.erode(thresh,None,iterations=4)
thresh = cv2.dilate(thresh,None,iterations=4)


#find contours and grab the largest
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)


# extracting extreme ponts in all 4 directions : tuple(c[][0])
exLeft = tuple(c[c[:,:,0].argmin()][0])
exRight = tuple(c[c[:,:,0].argmax()][0])
exTop = tuple(c[c[:,:,1].argmin()][0])
exBot = tuple(c[c[:,:,1].argmax()][0])


#draw the outline and mark all 4 points
cv2.drawContours(image,[c],-1,(0,255,200),2)
cv2.circle(image,exLeft , 8,( 0, 0, 255),-1)
cv2.circle(image,exRight , 8,(255 , 0, 0),-1)
cv2.circle(image, exTop, 8,( 0, 255, 0),-1)
cv2.circle(image, exBot, 8,( 200, 0, 255),-1)



cv2.imshow("image",image)
cv2.waitKey(0)
