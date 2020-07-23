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

def eval_f(test_imgs, test_imgs_idx, prediction, result_dir, train=False, result_dir_val=""):
    result_dir_backup = result_dir
    for (i, name) in enumerate(test_imgs_idx):
        if train:
            if i in indices_test:
                result_dir = result_dir_val
            else:
                result_dir = result_dir_backup
        w = test_imgs[i].shape[0]
        h = test_imgs[i].shape[1]
        patches_per_image = int(w*h/16/16)
        fr = patches_per_image*i
        to = patches_per_image*(i+1)
        labels = prediction[fr:to+1]
        predicted_img = label_to_img(w, h, patch_size, patch_size, labels)
        original_img = test_imgs[i]
        overlay = make_img_overlay(original_img, predicted_img)
        cimg = concatenate_images(original_img, predicted_img)

        img_number = int(re.search(r"\d+", test_imgs_idx[i]).group(0))

        predicted_img = img_float_to_uint8(predicted_img)
        cimg = img_float_to_uint8(cimg)
        prediction_dir = result_dir + "prediction/"
        concat_dir = result_dir + "concat/"
        overlay_dir = result_dir + "overlay/"
        Image.fromarray(predicted_img).save(prediction_dir + "prediction_" + str(img_number) + ".png")
        Image.fromarray(cimg).save(concat_dir + "concat_" + str(img_number) + ".png")
        overlay.save(overlay_dir + "overlay_" + str(img_number) + ".png")



def eval_unet(test_imgs, test_imgs_idx, prediction, result_dir, train=False, result_dir_val=""):
    result_dir_backup = result_dir
    for (i, name) in enumerate(test_imgs_idx):
        if train:
            if i in indices_test:
                result_dir = result_dir_val
            else:
                result_dir = result_dir_backup
        w = test_imgs[i].shape[0]
        h = test_imgs[i].shape[1]
        patches_per_image = int(w*h/16/16)
        fr = patches_per_image*i
        to = patches_per_image*(i+1)
        labels = prediction[fr:to+1]
        # print(to)
#         predicted_img = label_to_img(w, h, patch_size, patch_size, labels)
        predicted_img = label_to_img_unet(w, h, patch_size, patch_size, labels)
        original_img = test_imgs[i]
        overlay = make_img_overlay(original_img, predicted_img)
        cimg = concatenate_images(original_img, predicted_img)

        img_number = int(re.search(r"\d+", test_imgs_idx[i]).group(0))

        predicted_img = img_float_to_uint8(predicted_img)
        cimg = img_float_to_uint8(cimg)
        prediction_dir = result_dir + "prediction/"
        concat_dir = result_dir + "concat/"
        overlay_dir = result_dir + "overlay/"
        Image.fromarray(predicted_img).save(prediction_dir + "prediction_" + str(img_number) + ".png")
        Image.fromarray(cimg).save(concat_dir + "concat_" + str(img_number) + ".png")
        overlay.save(overlay_dir + "overlay_" + str(img_number) + ".png")


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

    ds_numpy = tfds.as_numpy(train_dataset) 
    num_images = 10 # 1900 # very very slow, trainset_size
    imgs = numpy.empty((num_images, 384, 384, 3))
    gt_imgs = numpy.empty((num_images, 384, 384, 1))
    for i, el in enumerate(ds_numpy): 
        if(i >= num_images): # otherwise the iterator runs forever
            break
        img, gt = el
        print(i)
        imgs[i] = img[0]
        gt_imgs[i] = gt[0]
     
    n = imgs.shape[0]
    img_patches = [img_crop_64(imgs[i], patch_size, patch_size) for i in range(n)]
    # unet
    gt_patches = [img_crop_64(gt_imgs[i], patch_size, patch_size) for i in range(n)]
    print("generated patches") # code above is extremely slow

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    test_dir = "/home/jonathan/CIL-street/data/test_images/"
    files = os.listdir(test_dir)
    print("Loading " + str(len(files)) + " test images")
    test_imgs = [load_image(test_dir + f) for f in files]

    patch_size = 16
    test_patches = [img_crop_64(test_imgs[i], patch_size, patch_size) for i in range(len(test_imgs))]
    test_patches = np.asarray([test_patches[i][j] / 255.0 for i in range(len(test_patches)) for j in range(len(test_patches[i]))])

    X = img_patches / 255.0
    Y = tf.where(gt_patches > 0, np.dtype('uint8').type(1), gt_patches)
    img_patches, gt_patches = None, None
    Y = np.expand_dims(Y, -1)
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, indices_train, indices_test = sk.train_test_split(X,Y,indices,test_size=0.20, random_state = 42)

    # taken from a blog post - which blog post?
    # -- Keras Functional API -- #
    # -- UNet Implementation -- #
    # Everything here is from tensorflow.keras.layers
    # I imported tensorflow.keras.layers * to make it easier to read

    input_size = (CROP_SIZE, CROP_SIZE, 3)

    # Contracting Path (encoding)
    inputs = Input(input_size)

    conv1 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # Expansive Path (decoding)
    up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1])
    conv9 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

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

    model = Sequential([
    #     Conv2D(16, 3, padding='same', activation='relu', input_shape=(patch_size, patch_size, 3)),
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
        ])

    model = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(64, 64, 3)))
    model.trainable = False
    pretrained_model = model

    model = tf.keras.Sequential([
        pretrained_model,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # from logits=True gives a strange error about shapes
                metrics=['accuracy'])

    print(model.summary())

    log_dir="logs/fit"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    filepath="weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    print("training model A")
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train,
            y_train,
            epochs=100,
            shuffle=True,
            validation_data=(X_test, y_test),
            callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                    ),
                    tensorboard_callback,
    #                 checkpoint
            ])

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=True,
        vertical_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) / 32, epochs=100,
            validation_data=(X_test, y_test),
            callbacks=[
                    keras.callbacks.EarlyStopping(
                        patience=5,
                        restore_best_weights=True,
                    ),
                    tensorboard_callback,
            ])

    data_gen_args = dict(
    # rotation_range=180.,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # fill_mode="constant",
    # cval=0,
    # horizontal_flip=True,
    # vertical_flip=True
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_generator = image_datagen.flow(X_train, batch_size=8, seed=seed)
    mask_generator = mask_datagen.flow(numpy.reshape(y_train, (4608, 64, 64, 1)), batch_size=8, seed=seed)

    aug = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
    ])

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        # HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                        #  val_shift_limit=10, p=.9),
        # CLAHE(p=1.0, clip_limit=2.0),
        # ShiftScaleRotate(
            # shift_limit=0.0625, scale_limit=0.1, 
            # rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
        # ToFloat(max_value=255)
    ])

    data_gen = Better_generator(image_generator, mask_generator, aug)

    TRAINSET_SIZE = X_train.shape[0]
    VALSET_SIZE = X_test.shape[0]
    EPOCHS = 50
    BATCH_SIZE = 16
    STEPS_PER_EPOCH = max(TRAINSET_SIZE // BATCH_SIZE, 1)
    VALIDATION_STEPS = max(VALSET_SIZE // BATCH_SIZE, 1)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    callbacks = [
        # to collect some useful metrics and visualize them in tensorboard
        tensorboard_callback,
        # if no accuracy improvements we can stop the training directly
        # tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        # to save checkpoints
        # tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
        # tf.keras.callbacks.ModelCheckpoint('best_model_unet64crop.{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
        # tf.keras.callbacks.ModelCheckpoint('best_model_unet64crop_aug.{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.h5', verbose=1, save_best_only=True, save_weights_only=True)
        tf.keras.callbacks.ModelCheckpoint('best_model_unet96crop_aug5.{epoch:03d}-{loss:.2f}-{val_loss:.2f}-{accuracy:.2f}-{val_accuracy:.2f}.h5', verbose=1, save_weights_only=True)
    ]

    # this does not work since train_generator is commented out
    '''model_history = model.fit(
        # X_train,
        # y_train,
        train_generator,
        # validation_split=0.1,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks)

    model.save('my_model.h5')'''


    #classification
    prediction = model.predict(test_patches, verbose=1)
    print(min(prediction), max(prediction))
    prediction = (np.sign(prediction)+1)/2
    print(min(prediction), max(prediction))

    prediction_train = model.predict(img_patches)
    prediction_train = (np.sign(prediction_train)+1)/2

    test_patches=None

    #unet
    prediction = model.predict(test_patches, verbose=1)
    print(prediction.shape)

    #unet
    prediction_train = model.predict(img_patches, verbose=1)
    print(prediction_train.shape)

    # # Display prediction as an image
    img_idx = 13
    w = gt_imgs[img_idx].shape[0]
    h = gt_imgs[img_idx].shape[1]
    patches_per_image = int(w*h/16/16)
    fr = patches_per_image*img_idx
    to = patches_per_image*(img_idx+1)
    labels = prediction_train[fr:to+1]
    predicted_im = label_to_img(w, h, patch_size, patch_size, labels)
    cimg = concatenate_images(imgs[img_idx], predicted_im)
    fig1 = plt.figure(figsize=(10, 10)) # create a figure with the default size 
    plt.imshow(cimg, cmap='Greys_r')

    result_dir = repo_dir + "code/patch_based/results_unet/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    create_res_dirs(result_dir + "test/")
    create_res_dirs(result_dir + "train/")
    create_res_dirs(result_dir + "val/")


    eval_f(test_imgs, test_imgs_idx, prediction, result_dir + "test/")

    eval_f(imgs, train_imgs_idx, prediction_train, result_dir + "train/", True, result_dir + "val/")

    prediction=None

    eval_unet(test_imgs, test_imgs_idx, prediction, result_dir + "test/")

    eval_unet(imgs, train_imgs_idx, prediction_train, result_dir + "train/", True, result_dir + "val/")
