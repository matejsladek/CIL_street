# -----------------------------------------------------------
# Collection of losses we tested.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.ndimage import distance_transform_edt as distance


def soft_dice_loss(y_true, y_pred):
    """
    Simple implementation of soft dice loss. See https://www.jeremyjordan.me/semantic-segmentation/ for more details
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    return 1-dice_coef(y_true, y_pred)


def surface_loss(y_true, y_pred):
    """
    Simple implementation of surface loss. See https://github.com/LIVIAETS/surface-loss for more details
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
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


def distance_map_loss(y_true, y_pred):
    """
    Custom loss based on computing the distance map of the groundtruth in order to weight misclassifications differently
    according to their distance from the road area.
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
    def dist_coeff(y_true):
        def calc_dist_coeff(y_true):
            y_true = y_true.numpy()
            negmask = ~(y_true.astype(np.bool))
            res = np.stack([distance(batch) for batch in negmask])
            max_dist = np.apply_over_axes(np.max, res, [1, 2])
            T = 0.05 * max_dist
            return np.where(res > T, T / max_dist, res / max_dist)
        dc = tf.py_function(func=calc_dist_coeff, inp=[y_true], Tout=tf.float32)
        return tf.math.exp(-dc)

    epsilon = K.epsilon()
    bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
    bce += dist_coeff(y_true) * (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
    return -K.mean(bce)


def focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        Simple implementation of surface loss. See https://arxiv.org/abs/1708.02002 for more details
        according to their distance from the road area.
        :param y_true: groundtruth
        :param y_pred: predictions
        :return: computed loss
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        epsilon = K.epsilon()
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed


def bce_diceloss(alpha=0.7):
    def fixed_bce_dice_loss(y_true, y_pred):
        """
        Mixed binary cross entropy and dice loss
        :param y_true: groundtruth
        :param y_pred: predictions
        :return: computed loss
        """
        epsilon = K.epsilon()
        bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
        bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
        bce = -bce
        return K.mean(alpha * bce + (1-alpha) * soft_dice_loss(y_true, y_pred))
    return fixed_bce_dice_loss()


def bce_logdice_loss(y_true, y_pred):
    """
    Mixed binary cross entropy and dice loss
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
    epsilon = K.epsilon()
    bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
    bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
    bce = -bce
    return K.mean(bce - tf.keras.backend.log(1. - soft_dice_loss(y_true, y_pred)))


def bce_surface_loss(y_true, y_pred):
    """
    Mixed binary cross entropy and surface loss
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
    epsilon = K.epsilon()
    bce = y_true * tf.math.log(K.clip(y_pred, epsilon, 1. - epsilon))
    bce += (1 - y_true) * tf.math.log(1 - K.clip(y_pred, epsilon, 1. - epsilon))
    return K.mean(-bce + surface_loss(y_true, y_pred))


def custom_loss(y_pred, y_true):
    """
    Binary cross entropy augmented to push predicted values towards 0 or 1. Works, but does not improve predictions.
    :param y_true: groundtruth
    :param y_pred: predictions
    :return: computed loss
    """
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + y_pred*(1-y_pred)