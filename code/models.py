import tensorflow as tf
from tensorflow.keras.layers import *
from classification_models.tfkeras import Classifiers

def PretrainedUnet(backbone_name='seresnext50', input_shape=(None, None, 3), encoder_weights='imagenet', encoder_freeze=False):

    decoder_filters=(256, 128, 64, 32, 16)
    n_blocks=len(decoder_filters)
    skip_layers_dict = {'seresnext50': (1078, 584, 254, 4), 'seresnext101': (2472, 584, 254, 4)}
    skip_layers = skip_layers_dict[backbone_name]

    backbone_fn, _ = Classifiers.get(backbone_name)
    backbone = backbone_fn(input_shape=input_shape, weights=encoder_weights, include_top=False)
    skips = ([backbone.get_layer(index=i).output for i in skip_layers])

    x = backbone.output
    for i in range(n_blocks):

        filters = decoder_filters[i]

        x = tf.keras.layers.UpSampling2D(size=2, name='decoder_stage{}_upsample'.format(i))(x)
        if i < len(skips):
            x = tf.keras.layers.Concatenate(axis=3, name='decoder_stage{}_concat'.format(i))([x, skips[i]])

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform', name='decoder_stage{}a_conv'.format(i))(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name='decoder_stage{}a_bn'.format(i))(x)
        x = tf.keras.layers.Activation('relu', name='decoder_stage{}a_activation'.format(i))(x)

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False, kernel_initializer='he_uniform', name='decoder_stage{}b_conv'.format(i))(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name='decoder_stage{}b_bn'.format(i))(x)
        x = tf.keras.layers.Activation('relu', name='decoder_stage{}b_activation'.format(i))(x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv')(x)
    x = tf.keras.layers.Activation('sigmoid', name='final_activation')(x)

    model = tf.keras.models.Model(backbone.input, x)

    if encoder_freeze:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    return model


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


# flexible implementation
def CustomUNet(blocks=4, conv_per_block=2, filters=16, activation='relu', dropout=0.2, bn=True, dilation=False, depth=6,
               aspp=False, aggregate='add', upsample=False):
    input_size = (400, 400, 3)
    inputs = Input(input_size)
    x = inputs
    skip = []
    for i in range(blocks):
        for j in range(conv_per_block):
            if bn and i + j > 0:
                x = BatchNormalization()(x)
            x = Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(x)
            x = Dropout(dropout)(x)
        skip.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        filters = filters * 2

    if dilation:
        dilated = []
        last_layer = None
        for i in range(depth):
            if aspp:
                if bn:
                    y = BatchNormalization()(y)
                y = Conv2D(filters, 3, activation=activation, dilation_rate=2 ** i, padding='same',
                           kernel_initializer='he_normal')(x)
                y = Dropout(dropout)(y)
                dilated.append(y)
                last_layer = y
            else:
                if bn:
                    x = BatchNormalization()(x)
                x = Conv2D(filters, 3, activation=activation, dilation_rate=2 ** i, padding='same',
                           kernel_initializer='he_normal')(x)
                x = Dropout(dropout)(x)
                dilated.append(x)
                last_layer = x
        if aggregate == 'add':
            x = tf.keras.layers.add(dilated)
        elif aggregate == 'concat':
            x = tf.keras.layers.concatenate(dilated)  # check axis
            x = Conv2D(filters, 1, activation=activation, padding='same', kernel_initializer='he_normal')(x)
        else:
            x = last_layer
    else:
        for i in range(conv_per_block):
            if bn:
                x = BatchNormalization()(x)
            x = Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(x)
            x = Dropout(dropout)(x)

    for i in range(blocks):
        filters = int(filters / 2)
        if upsample:
            x = UpSampling2D()(x)
        else:
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, skip[-(1 + i)]])
        for j in range(conv_per_block):
            if bn:
                x = BatchNormalization()(x)
            x = Conv2D(filters, 3, activation=activation, padding='same', kernel_initializer='he_normal')(x)
            x = Dropout(dropout)(x)

    x = Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


import segmentation_models as sm


def PretrainedNetOLD(backbone='vgg16', size=224, weights='imagenet', freeze=False):
    return sm.Unet(backbone, input_shape=(size, size, 3), encoder_weights=weights, encoder_freeze=freeze)
