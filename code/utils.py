import IPython.display as display
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
from PIL import Image
import os
import numpy as np

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def show_predictions(model=None, dataset=None, num=1):
    """Show a sample prediction.
    """
    for image, mask in dataset.take(num):
        pred_mask = model.predict(np.array(image))
        #print(pred_mask[0, 0, 0])
        #pred_mask = tf.argmax(pred_mask, axis=-1)
        #pred_mask = tf.expand_dims(pred_mask, axis=-1)
        display_sample([image[0], mask[0], pred_mask[0]])


def save_predictions(model, size=400, normalize=True, crop=True, name='output'):

  test_path = "CIL_street/data/test_images"
  test_list = os.listdir(test_path)
  test_list.sort()

  for image_path in test_list:
    print(image_path)
    if crop:
        image = np.array(Image.open(os.path.join(test_path, image_path)))
        img_parts = [image[:400, :400, :], image[:400, -400:, :], image[-400:, :400, :], image[-400:, -400:, :]]
        out_parts = []
        for img in img_parts:
            resized_img = np.array(Image.fromarray(img).resize((size,size)))
            resized_img = np.expand_dims(resized_img, 0)
            if normalize:
                resized_img = resized_img/255.0
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
        image = np.array(Image.open(os.path.join(test_path, image_path)).resize((size, size)))
        image = np.expand_dims(image, 0)
        if normalize:
            image = image/255.0
        output = model.predict(image)[0]
        if normalize:
            output = output * 255.0

    output = output.astype(np.uint8)
    output_img = tf.keras.preprocessing.image.array_to_img(output).resize((608, 608))

    output_img.save(os.path.join(('CIL_street/data/' + name), image_path))


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