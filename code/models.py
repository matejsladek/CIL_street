  # taken from a blog post
  # -- Keras Functional API -- #
  # -- UNet Implementation -- #
  # Everything here is from tensorflow.keras.layers
  # I imported tensorflow.keras.layers * to make it easier to read

import tensorflow as tf
from tensorflow.keras.layers import *

def UNet():
  input_size = (400, 400, 3)

  # Contracting Path (encoding)
  inputs = Input(input_size)

  conv1 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)
  conv1 = BatchNormalization()(conv1)
  conv1 = Dropout(0.1)(conv1)
  conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  conv1 = BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool1)
  conv2 = BatchNormalization()(conv2)
  conv2 = Dropout(0.1)(conv2)
  conv2 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv2)
  conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool2)
  conv3 = BatchNormalization()(conv3)
  conv3 = Dropout(0.2)(conv3)
  conv3 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv3)
  conv3 = BatchNormalization()(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool3)
  conv4 = BatchNormalization()(conv4)
  conv4 = Dropout(0.2)(conv4)
  conv4 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv4)
  conv4 = BatchNormalization()(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

  conv5 = Conv2D(256, 3, activation='elu', padding='same', kernel_initializer='he_normal')(pool4)
  conv5 = BatchNormalization()(conv5)
  conv5 = Dropout(0.3)(conv5)
  conv5 = Conv2D(256, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv5)
  conv5 = BatchNormalization()(conv5)

  # Expansive Path (decoding)
  up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
  merge6 = concatenate([up6, conv4])
  conv6 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge6)
  conv6 = BatchNormalization()(conv6)
  conv6 = Dropout(0.2)(conv6)
  conv6 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv6)
  conv6 = BatchNormalization()(conv6)

  up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
  merge7 = concatenate([up7, conv3])
  conv7 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge7)
  conv7 = BatchNormalization()(conv7)
  conv7 = Dropout(0.2)(conv7)
  conv7 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv7)
  conv7 = BatchNormalization()(conv7)

  up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
  merge8 = concatenate([up8, conv2])
  conv8 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge8)
  conv8 = BatchNormalization()(conv8)
  conv8 = Dropout(0.1)(conv8)
  conv8 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv8)
  conv8 = BatchNormalization()(conv8)

  up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
  merge9 = concatenate([up9, conv1])
  conv9 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(merge9)
  conv9 = BatchNormalization()(conv9)
  conv9 = Dropout(0.1)(conv9)
  conv9 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv9)
  conv9 = BatchNormalization()(conv9)

  conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

  return tf.keras.Model(inputs=inputs, outputs=conv10)