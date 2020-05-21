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


#implementatino from original paper
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


#flexible implementation
def CustomUNet(blocks=3, filters=32, activation='relu', dropout=0.2, bn=True, dilation=True, depth=6, aspp=True, aggregate='add'):

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
            x = Conv2D(filters, 1, activation, dilation_rate=2**i, padding='same', kernel_initializer='he_normal')(x)
        else:
            x = last_layer

        for i in range(blocks):
            filters = int(filters / 2)
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
            x = concatenate([x, skip[-i]])
            for j in range(2):
                x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
                if bn:
                    x = BatchNormalization()(x)
                x = Dropout(dropout)(x)

        x = Conv2D(1, 1, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=x)


#from efficientnet import EfficientNetB4

# taken from https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder#Training-Begins
def UnetPlusPlus(input_shape=(None, None, 3), dropout_rate=0.1):
    def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
        x = Conv2D(filters, size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation == True:
            x = LeakyReLU(alpha=0.1)(x)
        return x

    def residual_block(blockInput, num_filters=16):
        x = LeakyReLU(alpha=0.1)(blockInput)
        x = BatchNormalization()(x)
        blockInput = BatchNormalization()(blockInput)
        x = convolution_block(x, num_filters, (3, 3))
        x = convolution_block(x, num_filters, (3, 3), activation=False)
        x = Add()([x, blockInput])
        return x


    backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
    input = backbone.input
    start_neurons = 8

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same", name='conv_middle')(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    #     uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, deconv4_up1, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    #     uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    #     uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    #     uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    #     uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    return tf.keras.Model(input, output_layer)


from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# TODO: fix need for rescaling
# TODO: test
# TODO: also implement smaller resnets
# TODO: check input preprocessing
# based on https://www.kaggle.com/ishootlaser/cvrp-2018-starter-kernel-u-net-with-resnet50
def PretrainedUnet():
    R50 = ResNet50(include_top=False, weights='imagenet', input_shape=(416, 416, 3))
    R50.layers.pop()
    for layer in R50.layers:
        layer.trainable = False

    # Layers from ResNet50 to make skip connections
    skip_ix = [172, 140, 78, 36, 3]
    skip_end = []
    for i in skip_ix:
        skip_end.append(R50.layers[i])
    #R50.summary()

    for n, layer in enumerate(skip_end):
        n_channels = layer.output_shape[-1]
        print(layer.output_shape)
        if n == 0:
            concat_layer = layer.output
        else:
            prev_conv = UpSampling2D()(prev_conv)
            concat_layer = concatenate([layer.output, prev_conv])
        prev_conv = Conv2D(n_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat_layer)
        prev_conv = BatchNormalization()(prev_conv)
        prev_conv = Dropout(0.2)(prev_conv)
        prev_conv = Conv2D(n_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(prev_conv)
        prev_conv = BatchNormalization()(prev_conv)

    output = Conv2D(1, 1, activation='sigmoid')(prev_conv)

    return tf.keras.Model(inputs=R50.inputs, outputs=output)