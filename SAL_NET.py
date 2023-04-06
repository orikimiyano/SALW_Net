import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from loss_SAL import *

IMAGE_SIZE = 256
filter = 24


#####-----E-net------#####
# filters,kernel_size,strides
def initial_block(inp, nb_filter=13, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    max_pool = MaxPooling2D()(inp)
    merged = concatenate([conv, max_pool], axis=3)
    return merged

def bottleneck(inp, output, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = output // internal_scale
    encoder = inp
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise (Exception('You shouldn\'t be here'))

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    # 1x1
    encoder = Conv2D(output, (1, 1), use_bias=False)(encoder)

    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)

    other = inp
    # other branch
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)
        pad_feature_maps = output - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)

    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)

    return encoder

def en_build(inp, dropout_rate=0.01):
    enet = initial_block(inp)
    enet = BatchNormalization(momentum=0.1)(enet)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = PReLU(shared_axes=[1, 2])(enet)
    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        enet = bottleneck(enet, 64, dropout_rate=dropout_rate)  # bottleneck 1.i

    enet = bottleneck(enet, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enet = bottleneck(enet, 128)  # bottleneck 2.1
        enet = bottleneck(enet, 128, dilated=2)  # bottleneck 2.2
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.3
        enet = bottleneck(enet, 128, dilated=4)  # bottleneck 2.4
        enet = bottleneck(enet, 128)  # bottleneck 2.5
        enet = bottleneck(enet, 128, dilated=8)  # bottleneck 2.6
        enet = bottleneck(enet, 128, asymmetric=5)  # bottleneck 2.7
        enet = bottleneck(enet, 128, dilated=16)  # bottleneck 2.8

    return enet

# decoder
def de_bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder

def de_build(encoder, nc):
    enet = de_bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = de_bottleneck(enet, 64)  # bottleneck 4.1
    enet = de_bottleneck(enet, 64)  # bottleneck 4.2
    enet = de_bottleneck(enet, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    enet = de_bottleneck(enet, 16)  # bottleneck 5.1

    enet = Conv2DTranspose(filters=nc, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    return enet


def net(pretrained_weights=None, input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=20):
    inputs = Input(input_size)

    #####-----light U-net------#####

    # ConvBlock_1_64
    conv1 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)

    conv1 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv1)
    conv1 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)

    # ConvBlock_2_128
    conv2 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)

    conv2 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv2)
    conv2 = Conv2D(filter, 1, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    # ConvBlock_3_256
    conv3 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)

    conv3 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv3)
    conv3 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)

    # ConvBlock_4_512
    conv4 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)

    conv4 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv4)
    conv4 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)

    # ConvBlock_5_1024
    conv5 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)

    conv5 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv5)
    conv5 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)

    # upsampling_conv_4_512
    conv6 = Conv2D(filter * 4, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    conv6 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv6)
    conv6 = Conv2D(filter * 4, 1, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    # UpsamplingBlock_3_256
    up7 = (UpSampling2D(size=(2, 2))(conv6))

    # merge7 = concatenate([conv4, up7], axis=3)

    # upsampling_conv_3_256
    conv7 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)

    conv7 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv7)
    conv7 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    # upsampling_conv_2_128
    conv8 = Conv2D(filter * 2, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    conv8 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv8)
    conv8 = Conv2D(filter * 2, 1, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    # UpsamplingBlock_1_64
    up9 = (UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization()(up9)

    # merge9 = concatenate([conv2, up9], axis=3)

    # upsampling_conv_1_64
    conv9 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)

    conv9 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv9)
    conv9 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = LeakyReLU(alpha=0.3)(conv10)

    conv10 = DepthwiseConv2D(3, depth_multiplier=1, padding='same')(conv10)
    conv10 = Conv2D(filter, 3, padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = BatchNormalization()(conv10)

    unet_out = Conv2D(num_class, 1, activation='softmax', name='LUNET_out')(conv10)

    # loss_function = 'categorical_crossentropy'


    #####-----E-net------#####
    enet = en_build(inputs)
    enet = de_build(enet, num_class)
    enet_out = Conv2D(num_class, 1, activation='softmax', name='ENET_out')(enet)

    model = Model(inputs=inputs, outputs=[unet_out, enet_out])

    model.compile(loss={'LUNET_out': [fin_loss_multi],
                        'ENET_out': [multi_fin_loss]},
                  loss_weights={
                      'LUNET_out': 1.,
                      'ENET_out': 1.
                  },
                  metrics=['accuracy'])
    # model.summary()

    return model