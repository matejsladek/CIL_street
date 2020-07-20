# -----------------------------------------------------------
# Implementation of models. The second one is kept as a reference, while the first one is our best model.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import *
from classification_models.tfkeras import Classifiers
import logging


def slice_tensor(x, start, stop, axis):
    if axis == 3:
        return x[:, :, :, start:stop]
    elif axis == 1:
        return x[:, start:stop, :, :]
    else:
        raise ValueError("Slice axis should be in (1, 3), got {}.".format(axis))


def GroupConv2D(filters_in,filters_out,
                kernel_size,strides=(1, 1),groups=32,
                kernel_initializer='he_uniform',use_bias=True,
                activation='linear',padding='valid'):

    def layer(input_tensor):
        inp_ch = filters_in
        out_ch = filters_out
        slice_axis=3
        #inp_ch = int(backend.int_shape(input_tensor)[-1] // groups)  # input grouped channels
        #out_ch = int(filters // groups)  # output grouped channels

        blocks = []
        for c in range(groups):
            slice_arguments = {
                'start': c * inp_ch,
                'stop': (c + 1) * inp_ch,
                'axis': slice_axis,
            }
            x = Lambda(slice_tensor, arguments=slice_arguments)(input_tensor)
            x = Conv2D(out_ch,
                              kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_initializer,
                              use_bias=use_bias,
                              activation=activation,
                              padding=padding)(x)
            blocks.append(x)
        x = Concatenate(axis=slice_axis)(blocks)
        return x
    return layer


def __decoder_block_A(input_tensor,block_idx,decoder_filters,skips,residual,art,se):
    #3x3,3x3 with all possible res,art,se
    #close to best_model
    i = block_idx
    filters = decoder_filters[i]

    if art:
        cardinality = 32
        width = max(filters*2,cardinality)

    r = UpSampling2D(size=2, name='decoder_stage{}_upsample'.format(i))(input_tensor)
    # skip connection
    if i < len(skips):
        r = Concatenate(axis=3, name='decoder_stage{}_concat'.format(i))([r, skips[i]])

    x = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False,
            kernel_initializer='he_uniform', name='decoder_stage{}a_conv'.format(i))(r)
    x = BatchNormalization(axis=3, name='decoder_stage{}a_bn'.format(i))(x)
    x = Activation('relu', name='decoder_stage{}a_activation'.format(i))(x)

    # Squeeze and Excitation on the first convolution
    if se:
        w = GlobalAveragePooling2D(name='decoder_stage{}a_se_avgpool'.format(i))(x)
        w = Dense(filters // 8, activation='relu', name='decoder_stage{}a_se_dense1'.format(i))(w)
        w = Dense(filters, activation='sigmoid', name='decoder_stage{}a_se_dense2'.format(i))(w)
        x = Multiply(name='decoder_stage{}a_se_mult'.format(i))([x, w])

    if art:
        x = ZeroPadding2D(1)(x) #required due to custom GroupConv2D method
        x = GroupConv2D(filters_in=width//cardinality, filters_out=width//cardinality,
                kernel_size=(3, 3), strides=1, groups=cardinality,
                kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(axis=3, name='decoder_stage{}b_bn'.format(i))(x)
        x = Activation('relu', name='decoder_stage{}b_activation'.format(i))(x)

        x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False,
                kernel_initializer='he_uniform', name='decoder_stage{}c_conv'.format(i))(x)
        x = BatchNormalization(axis=3, name='decoder_stage{}c_bn'.format(i))(x)
        x = Activation('relu', name='decoder_stage{}c_activation'.format(i))(x)

    else:
        x = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False,
                kernel_initializer='he_uniform', name='decoder_stage{}b_conv'.format(i))(x)
        x = BatchNormalization(axis=3, name='decoder_stage{}b_bn'.format(i))(x)
        x = Activation('relu', name='decoder_stage{}b_activation'.format(i))(x)

    # Squeeze and Excitation on the second convolution
    if se:
        w = GlobalAveragePooling2D(name='decoder_stage{}b_se_avgpool'.format(i))(x)
        w = Dense(filters // 8, activation='relu', name='decoder_stage{}b_se_dense1'.format(i))(w)
        w = Dense(filters, activation='sigmoid', name='decoder_stage{}b_se_dense2'.format(i))(w)
        x = Multiply(name='decoder_stage{}b_se_mult'.format(i))([x, w])

    if residual:
        r = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False,
                kernel_initializer='he_uniform', name='decoder_stage{}sk_conv'.format(i))(r)
        r = BatchNormalization(axis=3, name='decoder_stage{}sk_bn'.format(i))(r)
        r = Activation('relu', name='decoder_stage{}sk_activation'.format(i))(r)
        x = Add(name='decoder_stage{}_add'.format(i))([x,r])

    return x


def __decoder_block_B(input_tensor,block_idx,decoder_filters,skips,residual,art,se):
    #1x1,3x3, 1x1 with all possible res,art,se
    i = block_idx
    filters = decoder_filters[i]
    mid_filters = filters//2

    if art:
        cardinality = 32
        width = max(mid_filters*2,cardinality)

    r = UpSampling2D(size=2, name='decoder_stage{}_upsample'.format(i))(input_tensor)
    # skip connection
    if i < len(skips):
        r = Concatenate(axis=3, name='decoder_stage{}_concat'.format(i))([r, skips[i]])

    x = Conv2D(filters=mid_filters, kernel_size=1, padding='same', use_bias=False,
            kernel_initializer='he_uniform', name='decoder_stage{}a_conv'.format(i))(r)
    x = BatchNormalization(axis=3, name='decoder_stage{}a_bn'.format(i))(x)
    x = Activation('relu', name='decoder_stage{}a_activation'.format(i))(x)

    if art:
        x = ZeroPadding2D(1)(x) #required due to custom GroupConv2D method
        x = GroupConv2D(filters_in=width//cardinality, filters_out=width//cardinality,
                kernel_size=(3, 3), strides=1, groups=cardinality,
                kernel_initializer='he_uniform', use_bias=False)(x)
        x = BatchNormalization(axis=3, name='decoder_stage{}b_bn'.format(i))(x)
        x = Activation('relu', name='decoder_stage{}b_activation'.format(i))(x)
    else:
        x = Conv2D(filters=mid_filters, kernel_size=3, padding='same', use_bias=False,
                kernel_initializer='he_uniform', name='decoder_stage{}b_conv'.format(i))(x)
        x = BatchNormalization(axis=3, name='decoder_stage{}b_bn'.format(i))(x)
        x = Activation('relu', name='decoder_stage{}b_activation'.format(i))(x)

    x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False,
            kernel_initializer='he_uniform', name='decoder_stage{}c_conv'.format(i))(x)
    x = BatchNormalization(axis=3, name='decoder_stage{}c_bn'.format(i))(x)
    x = Activation('relu', name='decoder_stage{}c_activation'.format(i))(x)

    if se:
        w = GlobalAveragePooling2D(name='decoder_stage{}a_se_avgpool'.format(i))(x)
        w = Dense(filters // 8, activation='relu', name='decoder_stage{}a_se_dense1'.format(i))(w)
        w = Dense(filters, activation='sigmoid', name='decoder_stage{}a_se_dense2'.format(i))(w)
        x = Multiply(name='decoder_stage{}a_se_mult'.format(i))([x, w])

    if residual:
        r = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False,
                kernel_initializer='he_uniform', name='decoder_stage{}sk_conv'.format(i))(r)
        r = BatchNormalization(axis=3, name='decoder_stage{}sk_bn'.format(i))(r)
        r = Activation('relu', name='decoder_stage{}sk_activation'.format(i))(r)
        x = Add(name='decoder_stage{}_add'.format(i))([x,r])

    return x


def RoadNet(backbone_name='seresnext50', input_shape=(None, None, 3), encoder_weights='imagenet',
            encoder_freeze=False, predict_distance=False, predict_contour=False,
            aspp=False, se=False, residual=False, art=False, 
            experimental_decoder=False,decoder_exp_setting=None):
    """
    Encoder-decoder based architecture for road segmentation in aerial images.

    :param backbone_name: name of the backbone network. Supported backbones are ResNet50, ResNet101, SEResNet50,
                          SEResNet101, ResNeXt50, ResNeXt101, SEResNeXt50 and  SEResNeXt101.
    :param input_shape: input shape, where the first two dimensions need to be a multiple of 16.
    :param encoder_weights: name of dataset for which to load weights. Only ImageNet is supported.
    :param encoder_freeze: freezes the weights in the backbone save from batch normalization layers
    :param predict_distance: if true, adds an additional output predicting the distance map of the road mask
    :param predict_contour: if true, adds an additional output predicting the contour of the road mask
    :param aspp: if true, the encoder output is passed through an ASPP module. More info at
                 http://liangchiehchen.com/projects/DeepLab.html
    :param se: if true, enables Squeeze and Excitation on the decoder convolutional blocks. More info at
               https://arxiv.org/abs/1709.01507
    :return: a tf.keras instance of the model
    """

    decoder_filters = (256, 128, 64, 32, 16)
    n_blocks = len(decoder_filters)
    skip_layers_dict = {'seresnext50': (1078, 584, 254, 4), 'seresnext101': (2472, 584, 254, 4),
                        'seresnet101': (552, 136, 62, 4), 'seresnet50': (246, 136, 62, 4),
                        'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
                        'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0')}
    skip_layers = skip_layers_dict[backbone_name]

    # load backbone network from external library
    backbone_fn, _ = Classifiers.get(backbone_name)
    backbone = backbone_fn(input_shape=input_shape, weights=encoder_weights, include_top=False)
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_layers])

    x = backbone.output

    # build ASPP if requested
    if aspp:
        b0 = GlobalAveragePooling2D()(x)
        b0 = Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(b0)
        b0 = Lambda(lambda x: tf.keras.backend.expand_dims(x, 1))(b0)
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp_pooling')(b0)
        b0 = BatchNormalization(name='aspp_pooling_bn')(b0)
        b0 = Activation('relu', name='aspp_pooling_relu')(b0)
        b0 = Lambda(lambda x : tf.image.resize(x, (12, 12)))(b0)

        b1 = Conv2D(256, 1, padding='same', dilation_rate=(1, 1), kernel_initializer='he_normal', name='aspp_b1_conv')(x)
        b1 = BatchNormalization(axis=3, name='aspp_b1_bn')(b1)
        b1 = Activation('relu', name='aspp_b1_relu')(b1)
        b2 = Conv2D(256, 3, padding='same', dilation_rate=(3, 3), kernel_initializer='he_normal', name='aspp_b2_conv')(x)
        b2 = BatchNormalization(axis=3, name='aspp_b2_bn')(b2)
        b2 = Activation('relu', name='aspp_b2_relu')(b2)
        b3 = Conv2D(256, 3, padding='same', dilation_rate=(6, 6), kernel_initializer='he_normal', name='aspp_b3_conv')(x)
        b3 = BatchNormalization(axis=3, name='aspp_b3_bn')(b3)
        b3 = Activation('relu', name='aspp_b3_relu')(b3)

        x = Concatenate(axis=3, name='aspp_concat')([b0, b1, b2, b3])
        x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp_concat_conv')(x)
        x = BatchNormalization(axis=3, name='aspp_concat_bn')(x)
        x = Activation('relu', name='aspp_concat_relu')(x)

    # create the decoder blocks sequentially
    if experimental_decoder:
        if decoder_exp_setting=="A":
            decoder_filters = (256, 128, 64, 32, 16)
            n_blocks = len(decoder_filters)
            for i in range(n_blocks):
                x = __decoder_block_A(x,i,decoder_filters,skips,
                        residual,art,se)
        elif decoder_exp_setting=="B":
            decoder_filters = (256, 128, 64, 32, 16)
            n_blocks = len(decoder_filters)
            for i in range(n_blocks):
                x = __decoder_block_B(x,i,decoder_filters,skips,
                        residual,art,se)
        elif decoder_exp_setting=="C":
            decoder_filters = (512, 256, 64, 32, 16)
            n_blocks = len(decoder_filters)
            for i in range(n_blocks):
                x = __decoder_block_A(x,i,decoder_filters,skips,
                        residual,art,se)
        elif decoder_exp_setting=="D":
            decoder_filters = (512, 256, 64, 32, 16)
            n_blocks = len(decoder_filters)
            for i in range(n_blocks):
                x = __decoder_block_B(x,i,decoder_filters,skips,
                        residual,art,se)
        else:
            raise(Exception)
    else:
        for i in range(n_blocks):
            x = __decoder_block_A(x,i,decoder_filters,skips,
                        residual,art,se)

    task1 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_mask')(x)
    task1 = Activation('sigmoid', name='final_activation_mask')(task1)

    # prepare for Multitask Learning
    if predict_contour:
        task2 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_contour')(x)
        task2 = Activation('sigmoid', name='final_activation_contour')(task2)
    if predict_distance:
        task3 = Conv2D(filters=1, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform', name='final_conv_distance')(x)
        task3 = Activation('linear', name='final_activation_distance')(task3)

    if predict_contour and predict_distance:
        output = [task1, task2, task3]
    elif predict_contour:
        output = [task1, task2]
    elif predict_distance:
        output = [task1, task3]
    else:
        output = task1

    model = tf.keras.models.Model(backbone.input, output)

    # freeze encoder weights if requested
    if encoder_freeze:
        for layer in backbone.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    return model


def CustomUNet(blocks=4, conv_per_block=2, filters=16, activation='relu', dropout=0.2, bn=True, dilation=False, depth=6,
               aspp=False, aggregate='add', upsample=False):
    """
    Flexible UNet implementation from our first experiments

    :param blocks: number of encoder and decoder blocks
    :param conv_per_block: number of convlutional layers in each block
    :param filters: filter in the first encoder block. Each block in the encoder has twice as many filters as the one
                    before, while the opposite happens in the decoder.
    :param activation: activation function
    :param dropout: dropout probability
    :param bn: activates Batch Normalization layers
    :param dilation: whether to use dilated convolutions in lowest blocks
    :param depth: number of convolutional layers in lowest block
    :param aspp: adds an ASPP module
    :param aggregate: selects how to aggregate the output of dilated convolutions (addition or concatenation)
    :param upsample: enables upsampling instead of transpose convolutions in the decoder
    :return: a tf.keras instance of the model
    """

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
