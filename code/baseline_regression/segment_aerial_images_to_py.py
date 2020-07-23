# -*- coding: utf-8 -*-
"""segment_aerial_images.ipynb
converted to a .py file
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
    return cimg, gt_img_3c

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

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
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


def run_experiment(config,prep_function):

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

    import os
    print(os.getcwd())

    # convert training images
    ds_numpy = tfds.as_numpy(train_dataset) 
    trainset_size = 10
    num_images = trainset_size # 1900 # very very slow
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

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    # Print feature statistics
    print('Computed ' + str(X.shape[0]) + ' features')
    print('Feature dimension = ' + str(X.shape[1]))
    print('Number of classes = ' + str(np.max(Y)))

    Y0 = [i for i, j in enumerate(Y) if j == 0]
    Y1 = [i for i, j in enumerate(Y) if j == 1]
    print('Class 0: ' + str(len(Y0)) + ' samples')
    print('Class 1: ' + str(len(Y1)) + ' samples')

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
    print('True positive rate = ' + str(TPR))

    # compute and save validation scores
    print('Saving validation scores')
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")
    out_file.write("Results of model.evaluate: \n")
    val_x = list(val_dataset_numpy_x)
    val_y = list(val_dataset_numpy_y)
    n = len(val_x)
    img_patches = [img_crop(val_x[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(val_y[i], patch_size, patch_size) for i in range(n)]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X_val = np.asarray([extract_features(img_patches[i]) for i in range(len(img_patches))])
    Y_val = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    model_evaluation = logreg.score(X_val, Y_val)
    out_file.write(str(model_evaluation))

    def kaggle_metric_simple(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(y_true == y_pred)/y_true.shape[0]

    out_file.write("\nKaggle metric on predictions: \n")
    predictions = logreg.predict(X_val) 
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
    test_imgs = [load_image(test_dir + test_files[i]) for i in range(len(test_files))]

    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)

    print(len(test_files))
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

    print('Finished ' + config['name'])


if __name__ == '__main__':
    # load each config file and run the experiment
    for config_file in glob('config/' + "*.json"):
        config = json.loads(open(config_file, 'r').read())
        name = config['name'] + '_' + datetime.datetime.now().strftime("%m%d_%H_%M_%S")
        config['log_folder'] = 'experiments/'+name
        os.makedirs(config['log_folder'])
        def prep_experiment():
            return 0
        run_experiment(config, prep_experiment)
