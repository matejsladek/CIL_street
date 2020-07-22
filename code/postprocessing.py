# -----------------------------------------------------------
# Implementation of our postprocessing methods.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import ndimage
from scipy.ndimage import measurements


def get_postprocess(name):
    """
    Retrieves the postprocessing method
    :param name: name of the postprocessing function
    :return: postprocessing method
    """
    if name == 'morphological':
        return morphological_postprocessing
    elif name == 'none':
        return no_postprocessing
    raise Exception('Unknown postprocessing')


def no_postprocessing(imgs):
    """
    Dummy postprocessing that does nothing.
    :param imgs: 3D numpy array of images to process
    :return: the same array
    """
    return imgs


def morphological_postprocessing(imgs, iterations=6):
    """
    Morphological transformation of imgs in order to binarize the image and remove noise through erosion and dilation.
    :param imgs: 3D numpy array of images to process
    :param iterations: number of iterations
    :return: postprocessed array
    """
    out = []
    original_shape = (imgs.shape[1], imgs.shape[2])
    for img in imgs:
        img = cv2.resize(img, (608, 608))
        kernel = np.ones((3,3), np.uint8)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, kernel, iterations=iterations)
        img = cv2.dilate(img, kernel, iterations=iterations)
        img = cv2.resize(img, original_shape)
        out.append(img)
    out = np.expand_dims(np.stack(out), -1)
    return out
