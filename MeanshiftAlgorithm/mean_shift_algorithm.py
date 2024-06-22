import cv2 as cv 
import numpy as np
#camera 
cap = cv.VideoCapture(0)
#yüzü tespit ettikten sonra takip için ilk önce frame okuması yapıyoruz
ret , frame = cap.read()

if ret == False:
    print("warning")

# detection 
face_cascade = cv.CascadeClassifier(r"C:\Users\cengh\Desktop\ComputerVison\MeanshiftAlgorithm\haarcascade_frontalface_default.xml")
face_rect =face_cascade.detectMultiScale(frame)
(face_x , face_y ,w ,h) = tuple(face_rect[0])
track_window = (face_x , face_y ,w ,h) #meanshift algoritması girdisi

#region of interest kutucuğun girdisi 

roi= frame[face_y:face_y+h, face_x:face_x+w] #roi =face 

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180]  )

cv.normalize(roi_hist, roi_hist,0,255,cv.NORM_MINMAX)


# takip için gerekli durdurma kriterleri
# eps değişiklik

term_crit = (cv.TermCriteria_EPS |cv.TermCriteria_COUNT, 5 , 1)

while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
        #histogramı bir görüntüde bulmak için kullanıyoruz
        #piksel karşılaştırma
        dst =cv.calcBackProject([hsv],[0], roi_hist, [0,180],1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h =track_window
        img2 = cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255,255),5)

        cv.imshow("meanshift",img2)
        if cv.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv.destroyAllWindows()