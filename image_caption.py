import urllib
import requests
import PIL
import glob
import shutil
import os
import string

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Embedding, Input
from tensorflow.python.keras.utils import to_categorical, plot_model
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import ModelCheckpoint

import cv2
import numpy as np

from os import listdir
from pickle import dump, load

# Extract
def feature_extraction(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        img_id = name.split('.')[0]
        features[img_id] = feature
        print('>%s' % name)
    return features

directory = 'Flicker8k_Dataset'

# Change directory and uncomment to extract features
# features = feature_extraction(directory)
# print('Features extracted: %d' % len(features))
# dump(features, open('features.pkl', 'wb'))

def load_the_documents(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_sets(filename):
    doc = load_the_documents(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)
    

filename = 'Flicker8k_text/Flickr8k.token.txt'
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

# Transform
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, description_list in descriptions.items():
        for i in range(len(description_list)):
            description = description_list[i]
            description = description.split()
            description = [word.lower() for word in description]
            description = [s.translate(table) for s in description]
            description = [word for word in description if len(word)>1]
            description = [word for word in description if word.isalpha()]
            description_list[i] = ' '.join(description)

clean_descriptions(descriptions)

def build_vocabulary(description):
    all_descriptions = set()
    for key in descriptions.keys():
        [all_descriptions.update(d.split()) for d in descriptions[key]]
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

def load_cleaned_descriptions(filename, dataset):
    doc = load_the_documents(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_description = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq' + ' '.join(image_description) + 'endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
training = load_sets(filename)
training_descriptions = load_cleaned_descriptions('descriptions.txt', training)
training_features = load_features('features.pkl', training)

def convert_from_dict_to_list(descriptions):
    all_descriptions = list()
    for key in descriptions.keys():
        [all_descriptions.append(s) for s in descriptions[key]]
    return all_descriptions

def create_tokens(descriptions):
    lines = convert_from_dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = create_tokens(training_descriptions)
vocabulary_size = len(tokenizer.word_index) + 1
# print('Vocab size = %d' % vocabulary_size)

def max_length(descriptions):
    lines = convert_from_dict_to_list(descriptions)
    return max(len(d.split()) for d in lines)

def initiate_sequencing(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    for key, description_list in descriptions.items():
        for desc in description_list:
            sequencing = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(sequencing)):
                in_sequence, out_sequence = sequencing[:i], sequencing[i]
                in_sequence = pad_sequences([in_sequence], maxlen=max_length)[0]
                out_sequence = to_categorical([out_sequence], num_classes=vocabulary_size)[0]
                X1.append(photos[key][0])
                X2.append(in_sequence)
                y.append(out_sequence)
    return np.array(X1), np.array(X2), np.array(y)

# Model
def caption_model(vocabulary_size, max_length):
    inputs1 = Input(shape = (4096, ))
    feature_extraction1 = Dropout(0.5)(inputs1)
    feature_extraction2 = Dense(256, activation = 'relu')(feature_extraction1)
    inputs2 = Input(shape = (max_length, ))
    sequence_embedding1 = Embedding(vocabulary_size, 256, mask_zero = True)(inputs2)
    sequence_embedding2 = Dropout(0.5)(sequence_embedding1)
    sequence_embedding3 = LSTM(256)(sequence_embedding2)
    decoder1 = add([feature_extraction1, feature_extraction2])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocabulary_size, activation = 'softmax')(decoder2)
    model = Model(inputs = [inputs1, inputs2], outputs = outputs)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model