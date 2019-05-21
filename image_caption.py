import urllib
import requests
import PIL
import glob
import shutil
import os
import string

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

directory = 'Flicker8k_Dataset'
features = feature_extraction(directory)
print('Features extracted: %d' % len(features))
dump(features, open('features.pkl', 'wb'))

def load_the_documents(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

filename = 'Flickr8k_text/Flicker8k.token.txt'
document = load_the_documents(filename)

def load_image_descriptions(document):
    maps = dict()
    for line in document.split("\n"):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_description = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_description = ' '.join(image_description)
        if image_id not in maps:
            maps[image_id] = list()
        maps[image_id].append(image_description)
    return maps

descriptions = load_image_descriptions(document)

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, description_list in descriptions.items():
        for i in range(len(description_list)):
            description = description_list[i]
            description = description.split()
            description = [word.lower() for word in description]
            description = [s.translate() for s in description]
            description = [word for word in description if len(word)>1]
            description = [word for word in description if word.isalpha()]
            description_list[i] = ' '.join(description)

clean_descriptions(descriptions)

def build_vocabulary(description):
    all_descriptions = set()
    for key in descriptions.keys():
        [all_descriptions.update(d.split) for d in descriptions[key]]
    return all_descriptions

vocabulary = build_vocabulary(descriptions)

def save_descriptions(descriptions, filename):
    lines = list()
    for key, description_list in descriptions.items():
        for description in description_list:
            lines.append(key + ' ' + description)
    desc_data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(desc_data)
    file.close()

save_descriptions(descriptions, 'descriptions.txt')

