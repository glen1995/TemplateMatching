import numpy as np 
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/media/glenja/New Volume/opencv/images/ronaldo.jpg',0)
template = cv2.imread('ronaldoface.jpg',0)
w,h = template.shape[::-1]

#different methods in template matching
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    meths = eval(meth)
    res = cv2.matchTemplate(img,template,meths)
    min_value,max_value,min_loc,max_loc = cv2.minMaxLoc(res)
    if meth in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (int(top_left[0]+w),int(top_left[1]+h))
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    plt.subplot(121),plt.imshow(res),plt.title('result from template matching')
    plt.subplot(122),plt.imshow(img),plt.title('orginal result')
    plt.show()