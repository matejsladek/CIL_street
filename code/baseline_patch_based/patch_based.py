# -----------------------------------------------------------
# Baseline #2: Patch-based CNN, adapted from the notebook
# provided in the context aof the course.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
# Commented out IPython magic to ensure Python compatibility.
# helper functions from https://github.com/dalab/lecture_cil_public/blob/master/exercises/2019/ex11_old/segment_aerial_images.ipynb
# needs batch size 1

import os
import matplotlib.image as mpimg
import numpy as np
import logging
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
import tensorflow_datasets as tfds
from code.metrics import *

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


# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im


def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def kaggle_metric_patches(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = y_pred.astype(int)
    return np.sum(y_true == y_pred) / y_true.shape[0]


def run_experiment(config,prep_function):
    """
    Trains and evaluates a model before computing and saving test predictions, all according to the config file.
    :param config: config dictionary
    :param prep_function: data loader
    :return: nothing
    """

    # tensorflow setup
    autotune = tf.data.experimental.AUTOTUNE

    # retrieve datasets
    train_dataset, val_dataset, val_dataset_numpy, training_data_glob, val_data_glob = prep_function(config, autotune)
    trainset_size = len(training_data_glob)
    valset_size = len(val_data_glob)
    logging.info(f"Training dataset contains {trainset_size} images.")
    logging.info(f"Validation dataset contains {valset_size} images.")
    steps_per_epoch = max(trainset_size // config['batch_size'], 1)

    # train
    logging.info('Begin training')

    # convert training images
    ds_numpy = tfds.as_numpy(train_dataset)
    ds_numpy_val = tfds.as_numpy(val_dataset)
    # it is very slow
    imgs = np.empty((trainset_size, 384, 384, 3))
    gt_imgs = np.empty((trainset_size, 384, 384, 1))
    val_imgs = np.empty((valset_size, 384, 384, 3))
    val_gt_imgs = np.empty((valset_size, 384, 384, 1))
    for i, el in enumerate(ds_numpy):
        if i >= trainset_size: # otherwise the iterator runs forever
            break
        img, gt = el
        imgs[i] = img[0]
        gt_imgs[i] = gt[0]

    for i, el in enumerate(ds_numpy_val):
        if i >= valset_size: # otherwise the iterator runs forever
            break
        val_img, val_gt = el
        val_imgs[i] = val_img[0]
        val_gt_imgs[i] = val_gt[0]


    img_patches = [img_crop(imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(trainset_size)]
    gt_patches = [img_crop(gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(trainset_size)]
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    X_train = img_patches / 255.0
    Y_train = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    val_img_patches = [img_crop(val_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(valset_size)]
    val_gt_patches = [img_crop(val_gt_imgs[i], PATCH_SIZE, PATCH_SIZE) for i in range(valset_size)]
    val_img_patches = np.asarray([val_img_patches[i][j] for i in range(len(val_img_patches)) for j in range(len(val_img_patches[i]))])
    val_gt_patches = np.asarray([val_gt_patches[i][j] for i in range(len(val_gt_patches)) for j in range(len(val_gt_patches[i]))])
    X_val = val_img_patches / 255.0
    Y_val = np.asarray([value_to_class(np.mean(val_gt_patches[i])) for i in range(len(val_gt_patches))])

    input_size = (PATCH_SIZE, PATCH_SIZE, 3)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=input_size),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    model.fit(X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=30,
            shuffle=True,
            callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                    ),
            ])


    # compute and save validation scores
    logging.info('Saving validation scores')
    out_file = open(config['log_folder'] + "/validation_score.txt", "w")
    out_file.write("Validation results\n")

    out_file.write("Kaggle metric on predictions: \n")
    predictions = (np.sign(model.predict(X_val)) + 1) / 2
    predictions = np.squeeze(predictions)
    kaggle_simple = kaggle_metric_patches(Y_val, predictions)
    out_file.write(str(kaggle_simple)+'\n')

    out_file.write("Accuracy: \n")
    model_evaluation = model.evaluate(X_val, Y_val)[1] # evaluate returns loss, accuracy
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
    logging.info("Loading " + str(len(test_files)) + " test images")
    test_imgs = [load_image(test_dir + test_files[i]) for i in range(len(test_files))]

    # folder for predictions
    pred_test_path = os.path.join(config['log_folder'], "pred_test")
    os.mkdir(pred_test_path)

    for idx, name in enumerate(test_files):
        test_patches = img_crop(test_imgs[i], PATCH_SIZE, PATCH_SIZE)
        test_patches = np.asarray([test_patches[j] / 255.0 for j in range(len(test_patches))])
        Zi = model.predict(test_patches)
        w = test_imgs[0].shape[0]
        h = test_imgs[0].shape[1]
        predicted_im = label_to_img(w, h, PATCH_SIZE, PATCH_SIZE, Zi)
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(predicted_im)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        result_img = gt_img_3c
        Image.fromarray(result_img).save(pred_test_path + name)

    logging.info('Finished ' + config['name'])
