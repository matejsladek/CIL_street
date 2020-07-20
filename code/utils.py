# -----------------------------------------------------------
# Collection of utilities.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from PIL import Image
import os
import numpy as np
import matplotlib.image as mpimg
import re
import glob


def save_predictions(model, crop, input_path, output_path, postprocessed_output_path, config, postprocess=None):
    """
    Loads test images, computes and saves the model's predictions
    :param model: models to use for prediction (usually trained)
    :param crop: if true, instead of resizing test images, it crops them and aggregates the predictions
    :param input_path: path to test images
    :param output_path: path where to save predictions
    :param postprocessed_output_path: path were to save predictions after postprocessing
    :param config: experiment config for additional parameters
    :param postprocess: postprocessing function. If None, postprocessing is not applied.
    :return: nothing
    """
    model_size = config['img_resize'],
    output_size = config['img_size_test'],
    normalize = config['normalize'],

    test_list = glob.glob(input_path + '/*.png')

    for image_path in test_list:

        if crop:
            # load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.uint8)

            # normalize image
            if normalize:
                image = tf.cast(image, tf.float32) / 255.0

            # divide image in crops
            img_parts = [image[:400, :400, :], image[:400, -400:, :], image[-400:, :400, :], image[-400:, -400:, :]]
            out_parts = []
            for img in img_parts:
                # compute predicted mask
                resized_img = tf.image.resize(img, (384, 384)).numpy()
                resized_img = np.expand_dims(resized_img, 0)
                if config['predict_contour'] or config['predict_distance']:
                    output = model.predict(resized_img)[0][0]
                else:
                    output = model.predict(resized_img)[0]
                if normalize:
                    output = output * 255.0
                out_parts.append(np.array(tf.keras.preprocessing.image.array_to_img(output).resize((400, 400))))

            # aggregate predictions
            output = np.zeros((608, 608))
            output[:304, :304] = out_parts[0][:304, :304]
            output[:304, -304:] = out_parts[1][:304, -304:]
            output[-304:, :304] = out_parts[2][-304:, :304]
            output[-304:, -304:] = out_parts[3][-304:, -304:]
            output = np.expand_dims(output, -1)
        else:
            # load image
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            # resize image
            image = tf.image.resize(image, (384, 384))
            # normalize image
            if normalize:
                image = tf.cast(image, tf.float32) / 255.0
            image = image.numpy()
            image = np.expand_dims(image, 0)

            # compute predicted mask
            if config['predict_contour'] or config['predict_distance']:
                output = model.predict(image)[0][0]
            else:
                output = model.predict(image)[0]

            if normalize:
                output = output * 255.0

        output = output.astype(np.uint8)
        output_img = tf.keras.preprocessing.image.array_to_img(output).resize((608, 608))

        # save prediction
        image_name = os.path.basename(image_path)
        output_img.save(output_path + '/' + image_name)
        if postprocess is None:
            continue
        # apply postprocessing and save copy
        postprocessed_output = np.squeeze(postprocess(np.expand_dims(output, 0)), axis=0)
        postprocessed_output_img = tf.keras.preprocessing.image.array_to_img(postprocessed_output).resize((608, 608))
        postprocessed_output_img.save(postprocessed_output_path + '/' + image_name)


def unet_freeze_encoder(model):
    for x in model.layers:
        if x.name.startswith("decoder"):
            break
        x.trainable = False


def to_csv(path, submission_filename):
    """
    Adapts the default csv saving scripts to work directly on a folder with predictions.
    :param path: path to predictions
    :param submission_filename: name of the sumbission file
    :return: nothing
    """

    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

    # assign a label to a patch
    def patch_to_label(patch):
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(image_filename):
        """Reads a single image and outputs the strings that should go into the submission file"""
        img_number = int(re.search(r"\d+", image_filename[-11:]).group(0))
        im = mpimg.imread(image_filename)
        patch_size = 16
        for j in range(0, im.shape[1], patch_size):
            for i in range(0, im.shape[0], patch_size):
                patch = im[i:i + patch_size, j:j + patch_size]
                label = patch_to_label(patch)
                yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))

    def masks_to_submission(submission_filename, *image_filenames):
        """Converts images into a submission file"""
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            for fn in image_filenames[0:]:
                f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))

    image_filenames = []
    for image_filename in glob.glob(path + '/*.png'):
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
