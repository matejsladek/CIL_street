import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def soft_dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

from scipy.ndimage import distance_transform_edt as distance

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return tf.keras.backend.mean(multipled)


def focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed


def compound_loss(alpha=0.7):
    def fixed_compound_loss(y_true, y_pred):
        epsilon = K.epsilon()
        bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
        bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
        return alpha * bce + (1-alpha) * soft_dice_loss(y_true, y_pred)
    return fixed_compound_loss
