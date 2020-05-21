  # taken from a blog post
  # -- Keras Functional API -- #
  # -- UNet Implementation -- #
  # Everything here is from tensorflow.keras.layers
  # I imported tensorflow.keras.layers * to make it easier to read

import tensorflow as tf
from tensorflow.keras.layers import *


# the network we have been using so far
def UNet():
    input_size = (400, 400, 3)

    # Contracting Path (encoding)
    inputs = Input(input_size)

    conv1 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv1)
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


#implementation with channel count from original paper
def Biomedical_UNet():
    input_size = (400, 400, 3)

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([up9, conv1])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    return tf.keras.Model(inputs=inputs, outputs=conv10)


# flexible implementation
def CustomUNet(blocks=3, filters=32, activation='relu', dropout=0.2, bn=True, dilation=True, depth=6, aspp=True, aggregate='add', upsample=False):

    input_size = (400, 400, 3)
    inputs = Input(input_size)
    x = inputs
    skip = []
    for i in range(blocks):
        for j in range(2):
            x = Conv2D(filters, 3, activation, padding='same', kernel_initializer='he_normal')(x)
            if bn:
                x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        filters = filters * 2

    if dilation:
        dilated = []
        last_layer = None
        for i in range(depth):
            if aspp:
                y = Conv2D(filters, 3, activation, dilation_rate=2**i, padding='same', kernel_initializer='he_normal')(x)
                if bn:
                    y = BatchNormalization()(y)
                y = Dropout(dropout)(y)
                dilated.append(y)
                last_layer = y
            else:
                x = Conv2D(filters, 3, activation, dilation_rate=2 ** i, padding='same', kernel_initializer='he_normal')(x)
                if bn:
                    x = BatchNormalization()(x)
                x = Dropout(dropout)(x)
                dilated.append(x)
                last_layer = x
        if aggregate == 'add':
            x = tf.keras.layers.add(dilated)
        elif aggregate == 'concat':
            x = tf.keras.layers.concatenate(dilated)  # check axis
            x = Conv2D(filters, 1, activation, padding='same', kernel_initializer='he_normal')(x)
        else:
            x = last_layer

        for i in range(blocks):
            filters = int(filters / 2)
            if upsample:
                x = UpSampling2D()(x)
            else:
                x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
            x = concatenate([x, skip[-i]])
            for j in range(2):
                x = Conv2D(filters, 3, activation, padding='same', kernel_initializer='he_normal')(x)
                if bn:
                    x = BatchNormalization()(x)
                x = Dropout(dropout)(x)

        x = Conv2D(1, 1, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# TODO: fix need for rescaling
# TODO: test
# TODO: check input preprocessing
# can use image_segmentation library instead
# based on https://www.kaggle.com/ishootlaser/cvrp-2018-starter-kernel-u-net-with-resnet50
def PretrainedUnet(freeze_encoder=True, backbone='resnet50', activation='relu', b_n=True, dropout=0.2):
    if backbone == 'resnet50':
        backbone = ResNet50(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
        backbone.layers.pop()
        skip_ix = [172, 140, 78, 36, 3]
    elif backbone == 'mobilenetv2':
        # FIXME
        backbone = MobileNetV2(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
        backbone.layers.pop()
        skip_ix = [-1, 119, 57, 30, 12]
    else:
        raise NotImplementedError

    if freeze_encoder:
        for layer in backbone.layers:
            layer.trainable = False

    # Layers from ResNet50 to make skip connections
    skip_end = []
    for i in skip_ix:
        skip_end.append(backbone.layers[i])
    # backbone.summary()

    for n, layer in enumerate(skip_end):
        n_channels = layer.output_shape[-1]
        print(layer.output_shape)
        if n == 0:
            concat_layer = layer.output
        else:
            prev_conv = UpSampling2D()(prev_conv)
            concat_layer = concatenate([layer.output, prev_conv])
        prev_conv = Conv2D(n_channels, 3, activation=activation, padding='same', kernel_initializer='he_normal')(concat_layer)
        if b_n:
            prev_conv = BatchNormalization()(prev_conv)
        prev_conv = Dropout(dropout)(prev_conv)
        prev_conv = Conv2D(n_channels, 3, activation=activation, padding='same', kernel_initializer='he_normal')(prev_conv)
        if b_n:
            prev_conv = BatchNormalization()(prev_conv)

    output = Conv2D(1, 1, activation='sigmoid')(prev_conv)

    return tf.keras.Model(inputs=backbone.inputs, outputs=output)