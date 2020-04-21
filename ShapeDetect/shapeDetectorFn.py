import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self,c):
        shape="unidentified"
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,.04*peri,True)
        # 1-5% of original contour perimeter

        if len(approx) == 3:
            shape="triangle"

        elif len(approx) ==4:
            (x,y,w,h) = cv2.boundingRect(approx)
            ar = w/float(h)

            shape ="square" if ar>=.95 and ar<=1.05 else "rectangle"

        elif len(approx)==5:
            shape="pentagon"

        else :
            shape="circle"

        return shape
