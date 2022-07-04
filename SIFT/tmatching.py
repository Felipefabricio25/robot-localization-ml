import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('botanic_map.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('test_map2.png', 0)
template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)

height, width = template.shape[::]

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
plt.imshow(res, cmap='gray')
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + width, top_left[1] + height)
cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2) 

cv2.imshow("Matched image", img_rgb)
cv2.waitKey()
cv2.destroyAllWindows()