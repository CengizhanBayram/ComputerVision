import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("contour.jpg",0)
plt.figure(), plt.imshow(img,cmap="gray"),plt.axis("off")
cv2.findContours(img , cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)