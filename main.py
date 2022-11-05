import cv2 
import cv
import time
import imutils

vs = cv2.VideoCapture(0)
cv2.waitKey(1)

firstFrame = None
area = 500

while True:
    flag,img = vs.read()
    text = "Normal"
    img = imutils.resize(img, width=500)

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gaussianImg
        continue
    
    imgDif = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDif, 25, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Difference", imgDif)
    # cv2.imshow("Threshold", threshImg)

    threshImg = cv2.dilate(threshImg, None, iterations=2)
    # cv2.imshow("dilated", threshImg)

    cnts, h = cv2.findContours(threshImg, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cnts = imutils.grab_contours(cnts)

    # cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)

    # print(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
        text = "Moving Object Detected"
    print(text)
    # cv2.putText(img, text, (10, 20))
    cv2.imshow("Camera feed", img)
    key = cv2.waitKey(1)
    if(key == ord("q")):
        break

vs.release()
cv2.destroyAllWindows()