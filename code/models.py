import tensorflow as tf
from tensorflow.keras.layers import *
from classification_models.tfkeras import Classifiers


def PretrainedUnet(backbone_name='seresnext50', input_shape=(None, None, 3), encoder_weights='imagenet',
                   encoder_freeze=False, predict_distance=False, predict_contour=False, aspp=False):

    decoder_filters=(256, 128, 64, 32, 16)
    n_blocks=len(decoder_filters)
    skip_layers_dict = {'seresnext50': (1078, 584, 254, 4), 'seresnext101': (2472, 584, 254, 4),
                        'seresnet101': (552, 136, 62, 4), 'seresnet50': (246, 136, 62, 4),
                        'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')}
    skip_layers = skip_layers_dict[backbone_name]

    backbone_fn, _ = Classifiers.get(backbone_name)
    backbone = backbone_fn(input_shape=input_shape, weights=encoder_weights, include_top=False)
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_layers])

    x = backbone.output

    if aspp:
        b0 = GlobalAveragePooling2D()(x)
        b0 = Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(b0)
        b0 = Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(b0)
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp_pooling')(b0)
        b0 = BatchNormalization(name='aspp_pooling_bn')(b0)
        b0 = tf.keras.layers.Activation('relu', name='aspp_pooling_relu')(b0)
        b0 = Lambda(lambda x : tf.image.resize(x, (12, 12)))(b0)

        b1 = Conv2D(256, 1, padding='same', dilation_rate=(1, 1), kernel_initializer='he_normal', name='aspp_b1_conv')(x)
        b1 = tf.keras.layers.BatchNormalization(axis=3, name='aspp_b1_bn')(b1)
        b1 = tf.keras.layers.Activation('relu', name='aspp_b1_relu')(b1)
        b2 = Conv2D(256, 3, padding='same', dilation_rate=(3, 3), kernel_initializer='he_normal', name='aspp_b2_conv')(x)
        b2 = tf.keras.layers.BatchNormalization(axis=3, name='aspp_b2_bn')(b2)
        b2 = tf.keras.layers.Activation('relu', name='aspp_b2_relu')(b2)
        b3 = Conv2D(256, 3, padding='same', dilation_rate=(6, 6), kernel_initializer='he_normal', name='aspp_b3_conv')(x)
        b3 = tf.keras.layers.BatchNormalization(axis=3, name='aspp_b3_bn')(b3)
        b3 = tf.keras.layers.Activation('relu', name='aspp_b3_relu')(b3)

        x = tf.keras.layers.Concatenate(axis=3, name='aspp_concat')([b0, b1, b2, b3])
        x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp_concat_conv')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name='aspp_concat_bn')(x)
        x = tf.keras.layers.Activation('relu', name='aspp_concat_relu')(x)

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

    if predict_contour and predict_distance:
        task1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_mask')(x)
        task1 = tf.keras.layers.Activation('sigmoid', name='final_activation_mask')(task1)
        task2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_contour')(x)
        task2 = tf.keras.layers.Activation('sigmoid', name='final_activation_contour')(task2)
        task3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_distance')(x)
        task3 = tf.keras.layers.Activation('linear', name='final_activation_distance')(task3)
        output = [task1, task2, task3]
    elif predict_contour:
        task1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_mask')(x)
        task1 = tf.keras.layers.Activation('sigmoid', name='final_activation_mask')(task1)
        task2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_contour')(x)
        task2 = tf.keras.layers.Activation('sigmoid', name='final_activation_contour')(task2)
        output = [task1, task2]
    elif predict_distance:
        task1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_mask')(x)
        task1 = tf.keras.layers.Activation('sigmoid', name='final_activation_mask')(task1)
        task2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_distance')(x)
        task2 = tf.keras.layers.Activation('linear', name='final_activation_distance')(task2)
        output = [task1, task2]
    else:
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv')(x)
        output = tf.keras.layers.Activation('sigmoid', name='final_activation')(x)

    model = tf.keras.models.Model(backbone.input, output)

    if encoder_freeze:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    return model


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
