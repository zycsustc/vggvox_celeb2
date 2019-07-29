# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from keras.layers import Input
from scipy.io import loadmat
import resnet50 as res
from keras import backend as K
from kerad.models import load_model
import get_vggvox_input as vg_input
import json
import os


def vggvox_model_init(already_saved):
    #New version for both establishing or loading model
    if(already_saved = False):
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        input_shape = Input((512, None, 1), name='input')
        print('Start initializing model ...')
        vggvox_celeb2 = res.ResNet50(input_shape)
        vggvox_celeb2.summary()
        print('Start loading weights ...')
        vggvox_celeb2 = res.give_weight(vggvox_celeb2)
        print('Finished establishing VGGVox_celeb2 model')
        vggvox_celeb2.save('vggvox_celeb2.h5')
        print('Model saved as vggvox_celeb2.h5')
        
    if (already_saved = True):
        vggvox_celeb2 = load_model('vggvox_celeb2.h5')
        
    return vggvox_celeb2


def speaker_character(input_file_path, output_path, vggvox_celeb2):
    model_input = vg_input.give_vggvox_input(input_file_path)
    z = model_input.apply(lambda x: np.squeeze(vggvox_celeb2.predict(x.reshape(1,*x.shape,1)))) #z_vector should be a pandas Series type variable
    z_vector = z.as_matrix()
    save_file_name = 'z_vector.json'
    with open(save_file_name,'w') as file_obj:
        json.dump(z_vector.tolist(), file_obj)
    output_path_saved = output_path+save_file_name
    return output_path_saved


def speaker_single(input_file_path):
    model_input = vg_input.give_vggvox_input_simple(input_file_path)
    z = model_input.apply(lambda x: np.squeeze(vggvox_celeb2.predict(x.reshape(1,*x.shape,1)))) #z_vector should be a pandas Series type variable
    z_array = z.as_matrix()
    return z_array
