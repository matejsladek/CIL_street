import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt as distance


def soft_dice_loss(y_true, y_pred):
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return 1-dice_coef(y_true, y_pred)


def surface_loss(y_true, y_pred):
    def calc_dist_map_batch(y_true):
        def calc_dist_map(seg):
            res = np.zeros_like(seg)
            posmask = seg.astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
            return res
        y_true_numpy = y_true.numpy()
        return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)

    y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
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


def bce_diceloss(alpha=0.7):
    def fixed_bce_dice_loss(y_true, y_pred):
        epsilon = K.epsilon()
        bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
        bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
        return alpha * bce + (1-alpha) * soft_dice_loss(y_true, y_pred)
    return fixed_bce_dice_loss()


def bce_logdice_loss(y_true, y_pred):
    epsilon = K.epsilon()
    bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
    bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
    return bce - tf.keras.backend.log(1. - soft_dice_loss(y_true, y_pred))


def distance_map_loss(y_true, y_pred):
    def dist_coeff(y_true):
        def calc_dist_coeff(y_true):
            negmask = ~y_true.astype(np.bool)
            res = np.stack([distance(batch) for batch in negmask])
            max_dist = np.apply_over_axes(np.max, res, [1, 2])
            T = 0.3 * max_dist
            return np.where(res > T, T / max_dist, res / max_dist)
        dc = tf.py_function(func=calc_dist_coeff, inp=[y_true], Tout=tf.float32)
        return tf.math.exp(-dc)

    epsilon = K.epsilon()
    bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
    bce += dist_coeff(y_true) * (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
    return bce