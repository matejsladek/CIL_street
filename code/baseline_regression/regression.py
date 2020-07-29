# -*- coding: utf-8 -*-
"""regression.ipynb (previously segment_aerial_images.ipynb)
converted to a .py file and made nicer
"""

from sklearn import linear_model
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from glob import glob
import json
import datetime
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy 
from sklearn import linear_model
from code.metrics import *

# Extract patches from input images
patch_size = 16 # each patch is 16*16 pixels

# Helper functions

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


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


# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat


# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features(img_patches[i]) for i in range(len(img_patches))])
    return X


# Compute features for each image patch
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im


def kaggle_metric_regression(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred)/y_true.shape[0]


def run_experiment(config,prep_function):

    # tensorflow setup
    autotune = tf.data.experimental.AUTOTUNE

    # retrieve datasets
    train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob = prep_function(config, autotune)
    trainset_size = len(training_data_glob)
    valset_size = len(val_data_glob)
    val_dataset_numpy_x, val_dataset_numpy_y = val_dataset_numpy
    logging.info(f"Training dataset contains {trainset_size} images.")
    logging.info(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)

    # train
    logging.info('Begin training')

    # convert training images
    ds_numpy = tfds.as_numpy(train_dataset) 
    # with all images it is very slow
    imgs = numpy.empty((trainset_size, 384, 384, 3))
    gt_imgs = numpy.empty((trainset_size, 384, 384, 1))
    for i, el in enumerate(ds_numpy): 
        if(i >= trainset_size): # otherwise the iterator runs forever
            break
        img, gt = el
        imgs[i] = img[0]
        gt_imgs[i] = gt[0]

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(trainset_size)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(trainset_size)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    # Print feature statistics
    logging.info('Computed ' + str(X.shape[0]) + ' features')
    logging.info('Feature dimension = ' + str(X.shape[1]))
    logging.info('Number of classes = ' + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    logging.info('Class 0: ' + str(len(Y0)) + ' samples')
    logging.info('Class 1: ' + str(len(Y1)) + ' samples')

    # train a logistic regression classifier
    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    logreg.fit(X, Y)

    # Predict on the training set
    Z = logreg.predict(X)

    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(Z)[0]
    Yn = np.nonzero(Y)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    logging.info('True positive rate = ' + str(TPR))

    # compute and save validation scores
    logging.info('Saving validation scores')
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")

    # extract features and classes from validation data
    val_x = list(val_dataset_numpy_x)
    val_y = list(val_dataset_numpy_y)
    val_size = len(val_x)
    img_patches = [img_crop(val_x[i], patch_size, patch_size) for i in range(val_size)]
    gt_patches = [img_crop(val_y[i], patch_size, patch_size) for i in range(val_size)]
    img_patches = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    gt_patches =  [gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]
    X_val = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y_val = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    out_file.write("Kaggle metric on predictions: \n")
    predictions = logreg.predict(X_val) 
    kaggle_simple = kaggle_metric_regression(Y_val, predictions)
    out_file.write(str(kaggle_simple)+'\n')

    # for patches the kaggle metric is the same as the accuracy
    # we keep it as a sanity check
    out_file.write("Accuracy: \n") 
    model_evaluation = logreg.score(X_val, Y_val)
    out_file.write(str(model_evaluation)+'\n')

    # our metrics expect float values
    predictions = predictions.astype(float)
    Y_val = Y_val.astype(float)

    out_file.write("F1 score: \n")
    out_file.write(str(f1_m(predictions, Y_val).numpy())+'\n')

    out_file.write("IoU: \n") # Intersection-Over-Union
    out_file.write(str(iou(predictions, Y_val).numpy())+'\n')

    out_file.close()

    # save predictions on test images
    test_dir = "data/test_images/"
    test_files = os.listdir(test_dir)
    n_test = len(test_files)
    logging.info("Loading " + str(len(test_files)) + " test images")
    test_imgs = [load_image(test_dir + test_files[i]) for i in range(len(test_files))]

    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)

    for idx, name in enumerate(test_files):
        Xi = extract_img_features(test_dir + test_files[idx])
        Zi = logreg.predict(Xi)
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

    logging.info('Finished ' + config['name'])

