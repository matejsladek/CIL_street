# partially based on
# https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
import cv2
import numpy as np
import pylab as plt
from glob import glob
import argparse
import os
import progressbar
import pickle as pkl
from numpy.lib import stride_tricks
from skimage import feature
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import mahotas as mt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import PIL
from PIL import Image

image_dir = '/home/jonathan/cil-rs/training/training/images' # images
label_dir = '/home/jonathan/cil-rs/training/training/groundtruth' # groundtruth

print ('[INFO] Reading image data.')

filelist = glob(os.path.join(image_dir, '*.png')) #.jpg for generated images
print("got filelist")
print(filelist)
image_list = []
label_list = []

for index, file in enumerate(filelist):
    image_list.append(cv2.imread(file, 1))
    gt_path = os.path.join(label_dir, os.path.basename(file).split('.')[0]+'.png') #.jpg for generated images
    label_list.append(cv2.imread(gt_path, 0))

print ('[INFO] Creating training dataset on %d image(s).' %len(image_list))

#for i, img in enumerate(image_list[0:]):
#   print(label_list)

img = image_list[0]
plt.imshow(img)

num_images = 100
image_size = 400
patch_size = 16
samples_per_image = (image_size//patch_size)**2 # 25*25 per image
# originally we have 100 * 400 * 400 * 3
# we want 100 * 25 * 25 * 16 * 16 * 3
X = np.array(image_list).reshape(num_images*samples_per_image, patch_size*patch_size*3)
# originally we have 100 * 400 * 400
# we want 100 * 25 * 25 (aggregate 16 * 16 pixels)
pil_img = Image.fromarray(label_list[0], mode='L')
pil_img.save("gt.jpg", "JPEG")
y_patches = np.array(label_list).reshape(num_images, image_size//patch_size, patch_size, image_size//patch_size, patch_size)
print(y_patches.shape)
for i in range(0, 24):
    for n in range(0, 24):
        pil_img = Image.fromarray(y_patches[0, i, :, n, :], mode='L')
        pil_img.save("patch"+str(i)+"_"+str(n)+".jpg", "JPEG")

y_agg = np.where(np.sum(y_patches, axis=(2, 4))/(patch_size**2) > 0.25, 1, 0)
print(y_agg[0])
print(y_agg.shape)
# doesnt seem to work
#img = Image.fromarray(y_agg[0]*255, mode='L')
#print(img)
#img.save("tiled.jpg", "JPEG")


y = y_agg.reshape(num_images*samples_per_image)
print(X.shape)
print(y.shape)

from sklearn.ensemble import GradientBoostingClassifier

# SVM is too slow, it does not even finish on PC
# from sklearn.svm import SVC
# print("calculating SVC")
# svm = SVC()
# svm = svm.fit(X, y)
# predicted = svm.predict(X)
# print(accuracy_score(y, predicted))

print("calculating GBC")
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gbc = gbc.fit(X, y)
predicted = gbc.predict(X)
print(accuracy_score(y, predicted))

print("calculating RFC")
# Initialize our model with 500 trees
rf = RandomForestClassifier(n_estimators=500, oob_score=True)

# Fit our model to training data
rf = rf.fit(X, y)

print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))


# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))

# Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2] - 1)

img_as_array = img[:, :, :7].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=img.shape,
                                        n=img_as_array.shape))

# Now predict for each pixel
class_prediction = rf.predict(img_as_array)

# Reshape our classification map
class_prediction = class_prediction.reshape(img[:, :, 0].shape)

# could also do all this in patches
'''
    patch_size = 16
    y_true = y_true.reshape(y_true.shape[0], int(y_true.shape[1]/patch_size), patch_size, int(y_true.shape[2]/patch_size), patch_size)
    y_pred = y_pred.reshape(y_true.shape[0], int(y_pred.shape[1]/patch_size), patch_size, int(y_pred.shape[2]/patch_size), patch_size)
    y_true = np.where(np.sum(y_true, axis=(2, 4))/(patch_size**2) > 0.25, 1, 0)
    y_pred = np.where(np.sum(y_pred, axis=(2, 4))/(patch_size**2) > 0.25, 1, 0)
    return np.array(np.sum(y_true == y_pred)/np.sum(np.ones_like(y_true))).astype(np.float32)
'''
