# -*- coding: utf-8 -*-
"""main.ipynb"""
# needs
# pip install -U git+https://github.com/albumentations-team/albumentations

# Commented out IPython magic to ensure Python compatibility.
# help functions from https://github.com/dalab/lecture_cil_public/blob/master/exercises/2019/ex11_old/segment_aerial_images.ipynb
# %matplotlib inline
from scipy.ndimage import rotate
import matplotlib.image as mpimg
import numpy as np
import re
import matplotlib.pyplot as plt
import os,sys
import datetime
from PIL import Image
from IPython.display import clear_output
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
import sklearn.model_selection as sk
import tensorflow_datasets as tfds
import numpy

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma,
    RGBShift,
    RandomContrast,
    RandomBrightness,
    HueSaturationValue,
    ShiftScaleRotate,
    ToFloat
)

import cv2

print(tf.__version__)

import os
from getpass import getpass
import urllib

CROP_SIZE = 64
PATCH_SIZE = 16
# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch


# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def img_crop_64(im, w, h):
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    if is_2d:
        im_padded = np.pad(im, ((CROP_SIZE,CROP_SIZE),(CROP_SIZE,CROP_SIZE)), 'constant')
    else:
        im_padded = np.pad(im, ((CROP_SIZE,CROP_SIZE),(CROP_SIZE,CROP_SIZE), (0,0)), 'constant')
    im = im_padded
    list_patches = []
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            num = (CROP_SIZE-PATCH_SIZE)//2
            if is_2d:
#                 im_patch = im[64+j-24:64+j+w+24, 64+i-24:64+i+h+24]
                im_patch = im[CROP_SIZE+j-num:CROP_SIZE+j+w+num, CROP_SIZE+i-num:CROP_SIZE+i+h+num]
            else:
                im_patch = im[CROP_SIZE+j-num:CROP_SIZE+j+w+num, CROP_SIZE+i-num:CROP_SIZE+i+h+num, :]
#                 print("3d", i, j, im_patch.shape)
            list_patches.append(im_patch)
            # list_patches.append(rotate(im_patch, 90))
            # list_patches.append(rotate(im_patch, 180))
            # list_patches.append(rotate(im_patch, 270))
    return list_patches


# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def label_to_img_unet(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            num = (CROP_SIZE-PATCH_SIZE)//2
            patch = labels[idx][num:CROP_SIZE-num, num:CROP_SIZE-num, :]
            im[j:j+w, i:i+h] = patch[:, :, 0]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


# unet compile
def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask'] * 100

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


class Better_generator:
    def __init__(self, im_gen, mask_gen, aug):
        self.im_gen, self.mask_gen, self.aug = im_gen, mask_gen, aug
    def __iter__(self):
        while True:
            nxt_img = self.im_gen.next()
            nxt_mask = self.mask_gen.next()
            img = []
            mask = []
            for (x, y) in zip(nxt_img, nxt_mask):
              augmented = self.aug(image=x, mask=y)
              img.append(augmented["image"])
              mask.append(augmented["mask"])
            img = np.stack(img, axis=0)
            mask = np.stack(mask, axis=0)
            # print(img.shape, mask.shape)
            yield img, mask


def create_res_dirs(result_dir):
    prediction_dir = result_dir + "prediction/"
    concat_dir = result_dir + "concat/"
    overlay_dir = result_dir + "overlay/"
    for dr in [result_dir, prediction_dir, concat_dir, overlay_dir]:    
        if not os.path.exists(dr):
            os.makedirs(dr)


def run_experiment(config,prep_function):
    """
    Trains and evaluates a model before computing and saving test predictions, all according to the config file.
    :param config: config dictionary
    :param prep_function: data loader
    :return: nothing
    """

    # needs batch size 1

    # tensorflow setup
    autotune = tf.data.experimental.AUTOTUNE

    # retrieve datasets
    train_dataset, val_dataset, val_dataset_numpy,\
    trainset_size, valset_size, training_data_root, val_data_root = prep_function(config,autotune)
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
    print(f"Training dataset contains {trainset_size} images.")
    print(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)

    # train
    print('Begin training')

    # Extract patches from input images
    patch_size = 16 # each patch is 16*16 pixels

    # convert training images
    ds_numpy = tfds.as_numpy(train_dataset) 
    # for testing, with all images it is very slow
    # trainset_size = 10
    num_images = trainset_size
    n = trainset_size
    imgs = numpy.empty((num_images, 384, 384, 3))
    gt_imgs = numpy.empty((num_images, 384, 384, 1))
    for i, el in enumerate(ds_numpy): 
        if(i >= num_images): # otherwise the iterator runs forever
            break
        img, gt = el
        print(i)
        imgs[i] = img[0]
        gt_imgs[i] = gt[0]
     
    img_patches = [img_crop_64(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop_64(gt_imgs[i], patch_size, patch_size) for i in range(n)]
    print("generated patches") # code above is extremely slow

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X = img_patches / 255.0
    Y = tf.where(gt_patches > 0, np.dtype('uint8').type(1), gt_patches)
    Y = np.expand_dims(Y, -1)


    # taken from a blog post - which blog post?
    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    # Everything here is from tensorflow.keras.layers
    # I imported tensorflow.keras.layers * to make it easier to read

    input_size = (CROP_SIZE, CROP_SIZE, 3)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # from logits=True gives a strange error about shapes
                metrics=['accuracy'])

    print(model.summary())

    model.fit(X,
            Y,
            epochs=10, # 100 takes too long for testing, and gets killed on my PC
            shuffle=True,
            callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                    ),
            ])


    # compute and save validation scores
    print('Saving validation scores')
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")
    out_file.write("Results of model.evaluate: \n")
    val_x = list(val_dataset_numpy_x)
    val_y = list(val_dataset_numpy_y)
    n = len(val_x)

    img_patches = [img_crop_64(val_x[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop_64(val_y[i], patch_size, patch_size) for i in range(n)]
    print("generated patches") # code above is extremely slow
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X_val = img_patches / 255.0
    Y = tf.where(gt_patches > 0, np.dtype('uint8').type(1), gt_patches)
    Y_val = np.expand_dims(Y, -1)
    model_evaluation = model.evaluate(X_val, Y_val)
    out_file.write(str(model_evaluation))

    def kaggle_metric_simple(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(y_true == y_pred)/y_true.shape[0]

    out_file.write("\nKaggle metric on predictions: \n")
    predictions = model.predict(X_val) 
    print(n)
    print(len(X_val))
    print(len(predictions))
    kaggle_simple = kaggle_metric_simple(Y_val, predictions)
    out_file.write(str(kaggle_simple))

    # save predictions on test images
    test_dir = "data/test_images/"
    test_files = os.listdir(test_dir)
    n_test = len(test_files)
    print("Loading " + str(len(test_files)) + " test images")
    patch_size = 16
    test_imgs = [load_image(test_dir + test_files[i]) for i in range(len(test_files))]
    
    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)

    print(len(test_files))
    for idx, name in enumerate(test_files):
        test_patches = img_crop_64(test_imgs[i], patch_size, patch_size)
        test_patches = np.asarray([test_patches[j] / 255.0 for j in range(len(test_patches))])
        Zi = model.predict(test_patches) 
        w = test_imgs[0].shape[0]
        h = test_imgs[0].shape[1]
        predicted_im = label_to_img(w, h, patch_size, patch_size, Zi)
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(predicted_im)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        result_img = gt_img_3c
        Image.fromarray(result_img).save(pred_test_path + name)

    print('Finished ' + config['name'])
