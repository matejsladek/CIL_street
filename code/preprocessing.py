import tensorflow as tf
import numpy as np

def parse_image(img_path: str) -> dict:

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    mask_path = tf.strings.regex_replace(img_path, "images", "groundtruth")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.where(mask > 0, np.dtype('uint8').type(1), mask)

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.
    """

    input_image = tf.image.resize(datapoint['image'], (400, 400))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (400, 400))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (400, 400))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (400, 400))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask
