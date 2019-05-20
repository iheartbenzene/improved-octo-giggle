import urllib
import requests
import PIL
import glob
import shutil
import os

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.python.keras.optimizers import SGD, Adadelta, Adagrad
from tensorflow.python.keras.utils import np_utils, generic_utils
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.advanced_activations import PReLU, LeakyReLU
from tensorflow.python.keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, Merge
from tensorflow.python.keras.preprocessing.text import one_hot
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

import cv2
import numpy as np

from os import listdir
from pickle import dump

def feature_extraction(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(256, 256))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        img_id = name.split('.')[0]
        features[img_id] = feature
        print('>%s' % name)
    return features

# set directory to begin
directory = 'Flicker8k_Dataset'
features = feature_extraction(directory)
print('Features extracted: %d' % len(features))
dump(features, open('features.pkl', 'wb'))