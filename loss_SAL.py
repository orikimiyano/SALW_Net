from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf

smooth = 1.
'''
Self-adjusting module
'''


def area_l(true, seg):
    pos_g = K.flatten(true)
    pos_p = K.flatten(seg)
    mul_p_g = pos_g * pos_p
    area_size = K.sum(pos_g - mul_p_g) + K.sum(pos_p - mul_p_g)

    return area_size


'''
loss for LUNET
'''


# tversky_loss

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


# FP item
def item_FP(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    gamma = 2
    precision = (true_pos + smooth) / (true_pos + false_pos + smooth)
    return K.pow((1 - precision), (1 / gamma))


# # FN item
# def item_FN(y_true, y_pred):
#     y_true_pos = K.flatten(y_true)
#     y_pred_pos = K.flatten(y_pred)
#     true_pos = K.sum(y_true_pos * y_pred_pos)
#     false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
#     gamma = 1
#     recall = (true_pos + smooth) / (true_pos + false_neg + smooth)
#     return K.pow((1 - recall), (1 / gamma))


# Optimal combination
def fin_loss(y_true, y_pred):
    loss1 = Cross_entropy_loss(y_true, y_pred)
    loss2 = item_FP(y_true, y_pred)
    # loss3 = item_FN(y_true, y_pred)
    rho = 0.7
    sigma_1 = 1
    # sigma_2 = 0

    # return (rho * loss1) + (1 - rho) * ((sigma_1 * loss2) + (sigma_2 * loss3))
    return (rho * loss1) + (1 - rho) * (sigma_1 * loss2)

def fin_loss_multi(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_loss = 0
    wl = 0

    for i in range(y_pred_n.shape[1]):
        single_loss = fin_loss(y_true_n[:, i], y_pred_n[:, i])
        num_all_int = tf.size(y_true_n[:, i])
        num_pos_int = tf.math.count_nonzero(y_true_n[:, i])

        num_pos = tf.cast(num_pos_int, dtype=tf.float32)
        num_all = tf.cast(num_all_int, dtype=tf.float32)
        wl = ((num_all - num_pos + smooth) / num_all + smooth) ** 1
        single_loss = wl * single_loss
        total_loss += single_loss
    area_v = area_l(y_true_n[:, i], y_pred_n[:, i])
    alph = tf.cast(area_v, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
    alph = -(alph - 1) ** 4 + 10 / 7
    alph = alph * 0.7
    total_loss = alph * total_loss/10000
    return total_loss


'''
loss for ENET
'''


def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.math.log(y_pred)
    return tf.reduce_sum(crossEntropyLoss, -1)


def multi_fin_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.
    for i in range(y_pred_n.shape[1]):
        single_loss = Cross_entropy_loss(y_true_n[:, i], y_pred_n[:, i])
        total_single_loss += single_loss

    area_v = area_l(y_true_n[:, i], y_pred_n[:, i])
    alph = tf.cast(area_v, dtype=tf.float32) / tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
    alph = -(alph - 1) ** 4 + 10 / 7
    alph = alph * 0.7
    total_single_loss = (1 - alph) * total_single_loss/10000

    return total_single_loss
