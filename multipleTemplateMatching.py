import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('coin.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('square.png',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

for i in zip(loc[1],loc[0]):
    cv2.rectangle(img_gray,i,(i[0]+w,i[1]+h),255,2)
cv2.imshow('fr',img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

