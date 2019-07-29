# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.models import Model
import os
import numpy as np
import pandas as pd
from keras import layers
from scipy.io import loadmat
from matlab_dict import mat_dict
#from . import get_submodules_from_kwargs
#from . import imagenet_utils
#from .imagenet_utils import decode_predictions
#from .imagenet_utils import _obtain_input_shape


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'conv' + str(stage) + '_' + str(block)
    bn_name_base = 'bn' + str(stage) + '_' + str(block)

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_1')(input_tensor)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_2')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_2')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_3')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_3')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'conv' + str(stage) + '_' + str(block)
    bn_name_base = 'bn' + str(stage) + '_' + str(block)

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_1')(input_tensor)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_2')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_2')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base+'_3')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1,name=bn_name_base+'_3')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name='relu_'+str(stage))(input_tensor)
    shortcut = layers.BatchNormalization(epsilon=1e-5, momentum=1,
        name='relubn_'+str(stage))(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet50(input_shape):
    
    inp = input_shape
    
    x = layers.ZeroPadding2D(padding=(3, 3), name='pad0')(inp)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv0')(x)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=1, name='bn0')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = layers.AveragePooling2D(pool_size = (8,3), strides = (1,1), name='pool_final')(x)
    x = layers.Conv2D(2048, kernel_size=(9,1), strides=(1,1), name='fc65')(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling2D(pool_size = (1,14), strides = (1,1), name='pool_time')(x)

    model = Model(inp, x, name='resnet50')
    return model

def conv_load_weights(model, name):
    matname = find_mat_name(name)
    file_path = 'weights_mat/'
    
    weight_w = loadmat(file_path+matname+'_f_b1.mat')
    weight_w = weight_w['temp_file']
    weight_b = loadmat(file_path+matname+'_b_b1.mat')
    weight_b = weight_b['temp_file']
    
    conv_weight = [weight_w, np.squeeze(weight_b)]
    
    model.get_layer(name).set_weights(conv_weight)
    
    return model

def bn_load_weights(model, name):
    matname = find_mat_name(name)
    file_path = 'weights_mat/'
    
    weight_bg = loadmat(file_path+matname+'_g_b1.mat')
    weight_bg = weight_bg['temp_file']
    weight_bb = loadmat(file_path+matname+'_b_b1.mat')
    weight_bb = weight_bb['temp_file']
    weight_bm = loadmat(file_path+matname+'_m_b1.mat')
    weight_bm = weight_bm['temp_file']
    
    b1 = np.squeeze(weight_bg)
    b2 = np.squeeze(weight_bb)
    b3 = np.squeeze(weight_bm[:,0])
    b4 = np.squeeze(weight_bm[:,1])
    bn_weight = [b1,b2,b3,b4**2]
    
    model.get_layer(name).set_weights(bn_weight)
    
    return model

def find_mat_name(name):
    mat_name = mat_dict[name]
    return mat_name

def give_weight(model):
    conv_load_weights(model, 'conv0')
    bn_load_weights(model, 'bn0')
    conv_load_weights(model, 'relu_2')
    bn_load_weights(model, 'relubn_2')
    print('initial down')
    
    #load weights for conv2
    conv_load_weights(model, 'conv2_1_1')
    bn_load_weights(model, 'bn2_1_1')
    print('ohou')
    conv_load_weights(model, 'conv2_1_2')
    bn_load_weights(model, 'bn2_1_2')
    conv_load_weights(model, 'conv2_1_3')
    bn_load_weights(model, 'bn2_1_3')
    print('conv2_1 down')
    
    conv_load_weights(model, 'conv2_2_1')
    bn_load_weights(model, 'bn2_2_1')
    conv_load_weights(model, 'conv2_2_2')
    bn_load_weights(model, 'bn2_2_2')
    conv_load_weights(model, 'conv2_2_3')
    bn_load_weights(model, 'bn2_2_3')
    print('conv2_2 down')
    
    conv_load_weights(model, 'conv2_3_1')
    bn_load_weights(model, 'bn2_3_1')
    conv_load_weights(model, 'conv2_3_2')
    bn_load_weights(model, 'bn2_3_2')
    conv_load_weights(model, 'conv2_3_3')
    bn_load_weights(model, 'bn2_3_3')
    print('conv2_3 down')
    
    #load weights for relu12
    conv_load_weights(model, 'relu_3')
    bn_load_weights(model, 'relubn_3')
    print('relu12 down')
    
    #load weights for conv3
    conv_load_weights(model, 'conv3_1_1')
    bn_load_weights(model, 'bn3_1_1')
    conv_load_weights(model, 'conv3_1_2')
    bn_load_weights(model, 'bn3_1_2')
    conv_load_weights(model, 'conv3_1_3')
    bn_load_weights(model, 'bn3_1_3')
    print('conv3_1 down')
    
    conv_load_weights(model, 'conv3_2_1')
    bn_load_weights(model, 'bn3_2_1')
    conv_load_weights(model, 'conv3_2_2')
    bn_load_weights(model, 'bn3_2_2')
    conv_load_weights(model, 'conv3_2_3')
    bn_load_weights(model, 'bn3_2_3')
    print('conv3_2 down')
    
    conv_load_weights(model, 'conv3_3_1')
    bn_load_weights(model, 'bn3_3_1')
    conv_load_weights(model, 'conv3_3_2')
    bn_load_weights(model, 'bn3_3_2')
    conv_load_weights(model, 'conv3_3_3')
    bn_load_weights(model, 'bn3_3_3')
    print('conv3_3 down')
    
    conv_load_weights(model, 'conv3_4_1')
    bn_load_weights(model, 'bn3_4_1')
    conv_load_weights(model, 'conv3_4_2')
    bn_load_weights(model, 'bn3_4_2')
    conv_load_weights(model, 'conv3_4_3')
    bn_load_weights(model, 'bn3_4_3')
    print('conv3_4 done')
    
    #load weights for relu28
    conv_load_weights(model, 'relu_4')
    bn_load_weights(model, 'relubn_4')
    print('relu28 done')
    
    #load weights for conv4
    conv_load_weights(model, 'conv4_1_1')
    bn_load_weights(model, 'bn4_1_1')
    conv_load_weights(model, 'conv4_1_2')
    bn_load_weights(model, 'bn4_1_2')
    conv_load_weights(model, 'conv4_1_3')
    bn_load_weights(model, 'bn4_1_3')
    print('conv4_1 done')
    
    conv_load_weights(model, 'conv4_2_1')
    bn_load_weights(model, 'bn4_2_1')
    conv_load_weights(model, 'conv4_2_2')
    bn_load_weights(model, 'bn4_2_2')
    conv_load_weights(model, 'conv4_2_3')
    bn_load_weights(model, 'bn4_2_3')
    print('conv4_2 done')
    
    conv_load_weights(model, 'conv4_3_1')
    bn_load_weights(model, 'bn4_3_1')
    conv_load_weights(model, 'conv4_3_2')
    bn_load_weights(model, 'bn4_3_2')
    conv_load_weights(model, 'conv4_3_3')
    bn_load_weights(model, 'bn4_3_3')
    print('conv4_3 done')
    
    conv_load_weights(model, 'conv4_4_1')
    bn_load_weights(model, 'bn4_4_1')
    conv_load_weights(model, 'conv4_4_2')
    bn_load_weights(model, 'bn4_4_2')
    conv_load_weights(model, 'conv4_4_3')
    bn_load_weights(model, 'bn4_4_3')
    print('conv4_4 done')
    
    conv_load_weights(model, 'conv4_5_1')
    bn_load_weights(model, 'bn4_5_1')
    conv_load_weights(model, 'conv4_5_2')
    bn_load_weights(model, 'bn4_5_2')
    conv_load_weights(model, 'conv4_5_3')
    bn_load_weights(model, 'bn4_5_3')
    print('conv4_5 done')
    
    conv_load_weights(model, 'conv4_6_1')
    bn_load_weights(model, 'bn4_6_1')
    conv_load_weights(model, 'conv4_6_2')
    bn_load_weights(model, 'bn4_6_2')
    conv_load_weights(model, 'conv4_6_3')
    bn_load_weights(model, 'bn4_6_3')
    print('conv4_6 done')
    
    #load weights for relu52
    conv_load_weights(model, 'relu_5')
    bn_load_weights(model, 'relubn_5')
    print('relu52 done')
    
    #load weights for conv5
    conv_load_weights(model, 'conv5_1_1')
    bn_load_weights(model, 'bn5_1_1')
    conv_load_weights(model, 'conv5_1_2')
    bn_load_weights(model, 'bn5_1_2')
    conv_load_weights(model, 'conv5_1_3')
    bn_load_weights(model, 'bn5_1_3')
    print('conv5_1 done')
    
    conv_load_weights(model, 'conv5_2_1')
    bn_load_weights(model, 'bn5_2_1')
    conv_load_weights(model, 'conv5_2_2')
    bn_load_weights(model, 'bn5_2_2')
    conv_load_weights(model, 'conv5_2_3')
    bn_load_weights(model, 'bn5_2_3')
    print('conv5_2 done')
    
    conv_load_weights(model, 'conv5_3_1')
    bn_load_weights(model, 'bn5_3_1')
    conv_load_weights(model, 'conv5_3_2')
    bn_load_weights(model, 'bn5_3_2')
    conv_load_weights(model, 'conv5_3_3')
    bn_load_weights(model, 'bn5_3_3')
    print('conv5_3 done')
    
    conv_load_weights(model, 'fc65')
    print('fc65 done')
    
    return model



# NUM_FFT = 512
# input_shape = (512,1000,1)
# inp = Input(input_shape,name='input')
# model = ResNet50(inp)

# model.summary()
