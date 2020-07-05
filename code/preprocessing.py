import tensorflow as tf
import numpy as np
import cv2
from scipy import ndimage


def get_parse_image(hard=True):
    def parse_image(img_path: str) -> dict:
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


def get_load_image_train(size=400, normalize=True, h_flip=0.5, v_flip=0.5, rot=0.25, contrast=0, brightness=0,
                         predict_contour=False, predict_distance=False):
    @tf.function
    def load_image_train(datapoint: dict) -> tuple:

        input_image = tf.image.resize(datapoint['image'], (size, size))
        input_mask = tf.image.resize(datapoint['mask'], (size, size))

        if tf.random.uniform(()) < h_flip:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        if tf.random.uniform(()) < v_flip:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        if tf.random.uniform(()) < rot:
            input_image = tf.image.rot90(input_image, 1)
            input_mask = tf.image.rot90(input_mask, 1)
        elif tf.random.uniform(()) < rot * 2:
            input_image = tf.image.rot90(input_image, 2)
            input_mask = tf.image.rot90(input_mask, 2)
        elif tf.random.uniform(()) < rot * 3:
            input_image = tf.image.rot90(input_image, 3)
            input_mask = tf.image.rot90(input_mask, 3)

        if tf.random.uniform(()) < contrast:
            input_image = tf.image.random_contrast(input_image, 0, 0.2)
        if tf.random.uniform(()) < brightness:
            input_image = tf.image.random_brightness(input_image, 0.2)

        if normalize:
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0

        output = input_mask

        def gradient_py(x):
            kernel = np.ones((3, 3), np.uint8)
            return np.expand_dims(cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel, iterations=3), axis=-1)

        def distance_py(x):
            res = ndimage.distance_transform_edt(x).astype(np.float32)
            return res

        input_contour = tf.numpy_function(func=gradient_py, inp=[input_mask], Tout=tf.float32)
        input_distance = tf.numpy_function(func=distance_py, inp=[input_mask], Tout=tf.float32)
        input_mask.set_shape((size, size, 1))
        input_contour.set_shape((size, size, 1))
        input_distance.set_shape((size, size, 1))

        if predict_contour and predict_distance:
            output = (input_mask, input_contour, input_distance)
        elif predict_contour:
            output = (input_mask, input_contour)
        elif predict_distance:
            output = (input_mask, input_distance)

        return input_image, output

    return load_image_train


def get_load_image_val(size=400, normalize=True, predict_contour=False, predict_distance=False):
    @tf.function
    def load_image_val(datapoint: dict) -> tuple:
        """Normalize and resize a test image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (size, size))
        input_mask = tf.image.resize(datapoint['mask'], (size, size))

        if normalize:
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0

        output = input_mask

        def gradient_py(x):
            kernel = np.ones((3, 3), np.uint8)
            return np.expand_dims(cv2.morphologyEx(x, cv2.MORPH_GRADIENT, kernel, iterations=3), axis=-1)

        def distance_py(x):
            res = ndimage.distance_transform_edt(x).astype(np.float32)
            return res

        input_contour = tf.numpy_function(func=gradient_py, inp=[input_mask], Tout=tf.float32)
        input_distance = tf.numpy_function(func=distance_py, inp=[input_mask], Tout=tf.float32)
        input_mask.set_shape((size, size, 1))
        input_contour.set_shape((size, size, 1))
        input_distance.set_shape((size, size, 1))

        if predict_contour and predict_distance:
            output = (input_mask, input_contour, input_distance)
        elif predict_contour:
            output = (input_mask, input_contour)
        elif predict_distance:
            output = (input_mask, input_distance)

        return input_image, output

    return load_image_val
