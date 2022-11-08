import cv2
import imutils

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

vs = cv2.VideoCapture(0)

while True:
    flag,img = vs.read()
    # text = "Normal"
    # img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('img',img)
    key = cv2.waitKey(1)
    if(key == ord("q")):
        break

vs.release()
cv2.destroyAllWindows()