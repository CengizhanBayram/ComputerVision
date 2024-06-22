import cv2 as cv 
import numpy as np

cap = cv.VideoCapture(0)

ret , frame = cap.read()


if ret == False:
    print("warning")

face