# -----------------------------------------------------------
# Implementation of our preprocessing methods.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import tensorflow as tf
import numpy as np
import cv2
from scipy import ndimage


def get_parse_image(hard=True):
    """
    Wrapper for image parsing method
    :param hard: flag for hard discretization of the mask
    :return: parsing method
    """
    def parse_image(img_path: str) -> dict:
        """
        Reads the image-mask pair at the specified path. Expects the mask to be in a sister folder to the image.
        If specified, binarizes the mask.
        :param img_path: path to the image
        :return: dictionary containing the image and its mask as tensors
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        mask_path = tf.strings.regex_replace(img_path, "images", "groundtruth")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        if hard:
            mask = tf.where(mask > 128, np.dtype('uint8').type(255), np.dtype('uint8').type(0))

        return {'image': image, 'mask': mask}
    return parse_image


def get_load_image_train(size=400, normalize=True, h_flip=0.5, v_flip=0.5, rot=0.25, contrast=0.3, brightness=0.1):
    """
    Returns the method to preprocess training data.
    :param size: size for data resizing
    :param normalize: enables normalization
    :param h_flip: probability of horizontally flipping each image
    :param v_flip: probability of vertically flipping each image
    :param rot: probability of rotating the image at a precise right angle. Total rotation probability is 3 times as much.
    :param contrast: parameter for random contrast augmentation
    :param brightness: parameter for random brightness augmentation
    :return: preprocessing method
    """
    @tf.function
    def load_image_train(datapoint: dict) -> tuple:
        """
        Applies preprocessing to the image-mask pair.
        :param datapoint: dict of tensors representing image and mask
        :return: tuple of preprocessed input image and output(s)
        """

        # resize images
        input_image = tf.image.resize(datapoint['image'], (size, size))
        input_mask = tf.image.resize(datapoint['mask'], (size, size))

        # flip images
        if tf.random.uniform(()) < h_flip:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
        if tf.random.uniform(()) < v_flip:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        # rotate images
        if tf.random.uniform(()) < rot:
            input_image = tf.image.rot90(input_image, 1)
            input_mask = tf.image.rot90(input_mask, 1)
        elif tf.random.uniform(()) < rot * 2:
            input_image = tf.image.rot90(input_image, 2)
            input_mask = tf.image.rot90(input_mask, 2)
        elif tf.random.uniform(()) < rot * 3:
            input_image = tf.image.rot90(input_image, 3)
            input_mask = tf.image.rot90(input_mask, 3)

        # apply random contrast and brightness
        if contrast > 0:
            input_image = tf.image.random_contrast(input_image, 1-contrast, 1+contrast)
        if brightness > 0:
            input_image = tf.image.random_brightness(input_image, brightness)

        # normalize images
        if normalize:
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0

        input_mask.set_shape((size, size, 1))

        return input_image, input_mask

    return load_image_train


def get_load_image_val(size=400, normalize=True):
    """
    Returns the method to preprocess validation and test data. Does not need augmentations.
    :param size: size for data resizing
    :param normalize: enables normalization
    :return: preprocessing method
    """
    @tf.function
    def load_image_val(datapoint: dict) -> tuple:
        """
        Applies preprocessing to the image-mask pair.
        :param datapoint: dict of tensors representing image and mask
        :return: tuple of preprocessed input image and output(s)
        """
        # resize images
        input_image = tf.image.resize(datapoint['image'], (size, size))
        input_mask = tf.image.resize(datapoint['mask'], (size, size))

        # normalize images
        if normalize:
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0

        input_mask.set_shape((size, size, 1))

        return input_image, input_mask

    return load_image_val
