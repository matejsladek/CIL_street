# -----------------------------------------------------------
# Implementation of metrics.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def recall_m(y_true, y_pred):
    """
    Computes recall on the predictions
    """
    true_positives = K.sum(K.clip(K.round(y_true) * K.round(y_pred), 0, 1))
    possible_positives = K.sum(K.clip(K.round(y_true), 0, 1))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Computes precision on the predictions
    """
    true_positives = K.sum(K.clip(K.round(y_true) * K.round(y_pred), 0, 1))
    predicted_positives = K.sum(K.clip(K.round(y_pred), 0, 1))
    print(predicted_positives)
    print(predicted_positives.dtype)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Computes f1 score on the predictions
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def iou(y_true, y_pred):
    """
    Computes IoU on the predictions
    """
    true_positives = K.sum(K.clip(K.round(y_true) * K.round(y_pred), 0, 1))
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    false_positives = predicted_positives - true_positives
    false_negatives = K.sum(K.clip(K.round(y_true) * K.round(1 - y_pred), 0, 1))
    return true_positives / (true_positives + false_positives + false_negatives + K.epsilon())


def np_kaggle_metric(y_true, y_pred):
    """
    Vectorized implementation of patch wise accuracy, where each patch of 16x16 pixels is labeled as 1 if it reaches
    25% of the maximum brightness.
    """
    patch_size = 16
    y_true = y_true.reshape(y_true.shape[0], int(y_true.shape[1]/patch_size), patch_size, int(y_true.shape[2]/patch_size), patch_size)
    y_pred = y_pred.reshape(y_true.shape[0], int(y_pred.shape[1]/patch_size), patch_size, int(y_pred.shape[2]/patch_size), patch_size)
    y_true = np.where(np.sum(y_true, axis=(2, 4))/(patch_size**2) > 0.25, 1, 0)
    y_pred = np.where(np.sum(y_pred, axis=(2, 4))/(patch_size**2) > 0.25, 1, 0)
    return np.array(np.sum(y_true == y_pred)/np.sum(np.ones_like(y_true))).astype(np.float32)


def kaggle_metric(y_true, y_pred):
    """
    Tensorflow wrapper for the custom kaggle metric
    """
    return tf.numpy_function(np_kaggle_metric, [tf.image.resize(y_true, [400, 400]), tf.image.resize(y_pred, [400, 400])], tf.float32)
