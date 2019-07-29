# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob

from wav_reader import get_fft_spectrum
import constants as c

def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step)
	end_frame = int(max_sec*frames_per_sec)
	step_frame = int(step_sec*frames_per_sec)
	for i in range(0, end_frame+1, step_frame):
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets

def give_vggvox_input(input_file_path):
    buckets = build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)
    file = pd.read_csv(input_file_path, delimiter=",")
    output = file['filename'].apply(lambda x: get_fft_spectrum(x, buckets)) 
    return output


def give_vggvox_input_simple(input_file_path):
    buckets = build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)
    output = get_fft_spectrum(input_file_path, buckets)
    return output
