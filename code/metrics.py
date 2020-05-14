import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def patch_to_label(patch):
    df = np.mean(patch, axis = (1, 2))
    return np.where(df > 0.25, 1, 0)

def np_kaggle_metric(y_true, y_pred):
    patch_size = 16
    correct = 0
    patch_count = 0
    for i in range(0, y_true.shape[1], patch_size):
        for j in range(0, y_true.shape[2], patch_size):
            patch_true = y_true[:, i:i + patch_size, j:j + patch_size, 0]
            label_true = patch_to_label(patch_true)
            patch_pred = y_pred[:, i:i + patch_size, j:j + patch_size, 0]
            label_pred = patch_to_label(patch_pred)
            patch_count += np.sum(np.ones_like(label_true))
            correct += np.sum(label_true == label_pred)

    return np.array(correct/(patch_count)).astype(np.float32)

def kaggle_metric(y_true, y_pred):
  return tf.numpy_function(np_kaggle_metric, [y_true, y_pred], tf.float32)