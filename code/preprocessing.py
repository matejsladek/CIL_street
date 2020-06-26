import tensorflow as tf
import numpy as np


def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.rgb_to_hsv(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "groundtruth")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    # mask = tf.where(mask > 0, np.dtype('uint8').type(1), mask)

    return {'image': image, 'mask': mask}


def make_binary_mask(datapoint: dict) -> dict:
    datapoint['mask'] = tf.where(datapoint['mask'] > 0, np.dtype('uint8').type(1), datapoint['mask'])
    return datapoint


def get_load_image_train(size=400, normalize=True, h_flip=0.5, v_flip=0.5, rot=0.25, contrast=0, brightness=0):
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
            # TODO: try other normalizations

        return input_image, input_mask

    return load_image_train


def get_load_image_test(size=400, normalize=True):
    @tf.function
    def load_image_test(datapoint: dict) -> tuple:
        """Normalize and resize a test image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (size, size))
        input_mask = tf.image.resize(datapoint['mask'], (size, size))

        if normalize:
            input_image = tf.cast(input_image, tf.float32) / 255.0
            input_mask = tf.cast(input_mask, tf.float32) / 255.0
            # TODO: try other normalizations normalize(input_image, input_mask)

        return input_image, input_mask

    return load_image_test
