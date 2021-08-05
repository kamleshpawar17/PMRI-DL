from customLayers import InstanceNormalization
from keras.layers import Input, UpSampling2D, Cropping2D, ZeroPadding2D, concatenate, Conv2D, Activation, add, \
    MaxPooling2D
from keras.models import Model
import numpy as np
import keras.backend as K


def UpSampling2D_BN_Act(kSize, crop, outPad, concateInp, inp, interpolation='nearest'):
    C = UpSampling2D(size=kSize, interpolation=interpolation)(inp)
    C_Crop = Cropping2D(cropping=crop)(C)
    C_Zpad = ZeroPadding2D(padding=outPad)(C_Crop)
    C_Con = concatenate([C_Zpad, concateInp], axis=-1)
    return C_Con

def conv2D_IN_Act_v1(nFeature, kSize, kStride, inp, padding='same', dilation_rate=(1, 1)):
    C = Conv2D(nFeature, kernel_size=kSize, strides=kStride, padding=padding, dilation_rate=dilation_rate,
               activation='relu')(inp)
    C_BN = InstanceNormalization()(C)
    return C_BN

def Unet_enc_dec_v1(n_channel=32, nFeatIn=1, nFeatOut=1, H=256, W=256, k_size=3, isRes=False):
    # ---- Encoder ---- #
    inp = Input(shape=(H, W, nFeatIn))
    C1_1 = conv2D_IN_Act_v1(nFeature=n_channel, kSize=k_size, kStride=1, inp=inp, padding='same')
    C1_2 = conv2D_IN_Act_v1(nFeature=n_channel, kSize=k_size, kStride=1, inp=C1_1, padding='same')
    C1_2_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(C1_2)
    C2_1 = conv2D_IN_Act_v1(nFeature=2 * n_channel, kSize=k_size, kStride=1, inp=C1_2_pool, padding='same')
    C2_2 = conv2D_IN_Act_v1(nFeature=2 * n_channel, kSize=k_size, kStride=1, inp=C2_1, padding='same')
    C2_2_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(C2_2)
    C3_1 = conv2D_IN_Act_v1(nFeature=4 * n_channel, kSize=k_size, kStride=1, inp=C2_2_pool, padding='same')
    C3_2 = conv2D_IN_Act_v1(nFeature=4 * n_channel, kSize=k_size, kStride=1, inp=C3_1, padding='same')
    C3_2_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(C3_2)
    C4_1 = conv2D_IN_Act_v1(nFeature=8 * n_channel, kSize=k_size, kStride=1, inp=C3_2_pool, padding='same')
    C4_2 = conv2D_IN_Act_v1(nFeature=8 * n_channel, kSize=k_size, kStride=1, inp=C4_1, padding='same')
    C4_2_pool = MaxPooling2D(pool_size=(2, 2), strides=2)(C4_2)
    C5_1 = conv2D_IN_Act_v1(nFeature=8 * n_channel, kSize=k_size, kStride=1, inp=C4_2_pool, padding='same')
    C5_2 = conv2D_IN_Act_v1(nFeature=8 * n_channel, kSize=k_size, kStride=1, inp=C5_1, padding='same')

    # ---- Decoder ---- #
    C4_2_us = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=C4_2, inp=C5_2, interpolation='bilinear')
    C4_1_d = conv2D_IN_Act_v1(nFeature=4 * n_channel, kSize=k_size, kStride=1, inp=C4_2_us, padding='same')
    C4_2_d = conv2D_IN_Act_v1(nFeature=4 * n_channel, kSize=k_size, kStride=1, inp=C4_1_d, padding='same')
    C3_2_us = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=C3_2, inp=C4_2_d, interpolation='bilinear')
    C3_1_d = conv2D_IN_Act_v1(nFeature=2 * n_channel, kSize=k_size, kStride=1, inp=C3_2_us, padding='same')
    C3_2_d = conv2D_IN_Act_v1(nFeature=2 * n_channel, kSize=k_size, kStride=1, inp=C3_1_d, padding='same')
    C2_2_us = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=C2_2, inp=C3_2_d, interpolation='bilinear')
    C2_1_d = conv2D_IN_Act_v1(nFeature=1 * n_channel, kSize=k_size, kStride=1, inp=C2_2_us, padding='same')
    C2_2_d = conv2D_IN_Act_v1(nFeature=1 * n_channel, kSize=k_size, kStride=1, inp=C2_1_d, padding='same')
    C1_2_us = UpSampling2D_BN_Act(kSize=2, crop=0, outPad=0, concateInp=C1_2, inp=C2_2_d, interpolation='bilinear')
    C1_1_d = conv2D_IN_Act_v1(nFeature=n_channel, kSize=k_size, kStride=1, inp=C1_2_us, padding='same')
    C1_2_d = conv2D_IN_Act_v1(nFeature=n_channel, kSize=k_size, kStride=1, inp=C1_1_d, padding='same')
    C1_3_d = conv2D_IN_Act_v1(nFeature=int(n_channel / 2), kSize=1, kStride=1, inp=C1_2_d, padding='same')
    conv_last = Conv2D(filters=nFeatOut, kernel_size=1, strides=1, padding='same')(C1_3_d)
    if isRes:
        conv_last = add([conv_last, inp])
    model = Model(inputs=inp, outputs=conv_last)
    return model