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


def display_sample(display_list):
    '''
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    '''
    pass


def show_predictions(model=None, dataset=None, num=1):
    """Show a sample prediction.
    """
    for image, mask in dataset.take(num):
        pred_mask = model.predict(np.array(image))
        display_sample([image[0], mask[0], pred_mask[0]])


def save_predictions(model, crop, input_path, output_path, postprocessed_output_path, config, postprocess=None):
    model_size = config['img_resize'],
    output_size = config['img_size_test'],
    normalize = config['normalize'],

    test_list = glob.glob(input_path + '/*.png')

    for image_path in test_list:
        print(image_path)
        if crop:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            if normalize:
                image = tf.cast(image, tf.float32) / 255.0

            img_parts = [image[:400, :400, :], image[:400, -400:, :], image[-400:, :400, :], image[-400:, -400:, :]]
            out_parts = []
            for img in img_parts:
                resized_img = tf.image.resize(img, (384, 384)).numpy()
                resized_img = np.expand_dims(resized_img, 0)
                if config['predict_contour'] or config['predict_distance']:
                    output = model.predict(resized_img)[0][0]
                else:
                    output = model.predict(resized_img)[0]
                if normalize:
                    output = output * 255.0
                out_parts.append(np.array(tf.keras.preprocessing.image.array_to_img(output).resize((400, 400))))

            output = np.zeros((608, 608))
            output[:304, :304] = out_parts[0][:304, :304]
            output[:304, -304:] = out_parts[1][:304, -304:]
            output[-304:, :304] = out_parts[2][-304:, :304]
            output[-304:, -304:] = out_parts[3][-304:, -304:]
            output = np.expand_dims(output, -1)
        else:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.uint8)
            image = tf.image.resize(image, (384, 384))
            if normalize:
                image = tf.cast(image, tf.float32) / 255.0
            image = image.numpy()
            image = np.expand_dims(image, 0)

            if config['predict_contour'] or config['predict_distance']:
                output = model.predict(image)[0][0]
            else:
                output = model.predict(image)[0]

            if normalize:
                output = output * 255.0

        output = output.astype(np.uint8)
        output_img = tf.keras.preprocessing.image.array_to_img(output).resize((608, 608))

        image_name = os.path.basename(image_path)
        output_img.save(output_path + '/' + image_name)
        if postprocess is None:
            return
        postprocessed_output = np.squeeze(postprocess(np.expand_dims(output, 0)), axis=0)
        postprocessed_output_img = tf.keras.preprocessing.image.array_to_img(postprocessed_output).resize((608, 608))
        postprocessed_output_img.save(postprocessed_output_path + '/' + image_name)


def plot_loss(model_history):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([np.min(val_loss + loss) - 0.1, np.max(val_loss + loss) + 0.1])
    plt.legend()
    plt.show()


def unet_freeze_encoder(model):
    for x in model.layers:
        if x.name.startswith("decoder"):
            break
        x.trainable = False


def to_csv(path, submission_filename):

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
