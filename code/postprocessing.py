import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

img_paths = glob.glob('output/*.png')
for img_path in img_paths:
    img = cv2.imread(img_path, 0)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=3)
    img = cv2.erode(img, kernel, iterations=8)
    img = cv2.dilate(img, kernel, iterations=3)
    cv2.imwrite(img_path.replace('output', '3x3_dilate3_erode8_dilate3'), img)
