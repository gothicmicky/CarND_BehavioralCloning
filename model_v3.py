import os
import pickle
import json
import random
import csv
import glob
import pandas as pd
import pickle

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from img_preprocess import img_preprocess
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.utils import np_utils
import time
import sys

_index_in_epoch = 0
batch_size = 256
img_height, img_width = 160, 320

####################################################################
def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

def load_training_and_validation_from_dataframe(df):
    rows, labels = [], []
    correction = 0.25 # this is a parameter to tune

    for idx in range(len(df)):
        angle = df['steering_angle'][idx]

        ## left image
        img_path = df['left_image'][idx]
        labels.append(angle + correction)
        rows.append(img_path)

        ## centre image
        img_path = df['center_image'][idx]
        labels.append(angle)
        rows.append(img_path)

        ## right image
        img_path = df['right_image'][idx]
        labels.append(angle - correction)
        rows.append(img_path)

    assert len(rows) == len(labels), 'unbalanced data'

    # shuffle the data
    X, Y = shuffle(np.array(rows), np.array(labels))

    # split into training and validation
    return train_test_split(X, Y, test_size = .25)
####################################################################

def next_batch(data, labels, batch_size):
    """
    Return the next `batch_size` examples from this data set.
    """
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        data, labels = shuffle(data, labels)
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]

####################################################################

def transform_generator(x, y, batch_size, is_validation=False):
    while True:
        bad = []
        images, labels = list(), list()
        _images, _labels = next_batch(x, y, batch_size)
        for i in range(len(_images)):
            img = cv2.imread('data2/'+_images[i])
            if img is None: continue
            else: bad.append('/home/sameh/short-p/data2/{}'.format(_images[i]))
            img = img.reshape(img_height, img_width, 3)
            angle = _labels[i]

            #pre-processing
            img, angle = img_preprocess(img, angle, is_validation)
            img = cv2.resize(img, (200,66))
            
            images.append(img)
            labels.append(angle)

        X = np.array(images, dtype=np.float64).reshape((-1, 66, 200, 3))
        Y = np.array(labels, dtype=np.float64)            
        yield (X, Y)

####################################################################

def gen_model(model_type = "nvidia"):
    model = Sequential()
    #data preprocessing: normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    if model_type == "lenet":
        #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
        model.add(Convolution2D(6,5,5,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(6,5,5,activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))

    if model_type == 'nvidia':  
        keep_prob = 0.5
        #model.add(Cropping2D(cropping=((65,25),(1,1))), input_shape=(img_height,img_width,3))
        model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(64,3,3,activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(64,3,3,activation='relu'))
        Dropout(keep_prob)
        model.add(Flatten())
        # layer 5, fc
        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dropout(keep_prob))
        # layer 6, fc
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(keep_prob))
        # layer 7, fc
        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        # layer 8, fc
        model.add(Dense(10))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        # layer output
        model.add(Dense(1))

    if model_type == 'nvidia_v2':
                # -------------------------------------
        # Cover of NVidia end-to-end network (widen conv layers)
        # -------------------------------------
        # layer 1, conv
        model.add(Convolution2D(36, 5, 5, subsample=(2,2), input_shape=(66, 200, 3)))
        model.add(Activation('relu'))
        # layer 2, conv
        model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # layer 3, conv
        model.add(Convolution2D(64, 5, 5, subsample=(2,2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # layer 4, conv
        model.add(Convolution2D(96, 3, 3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # layer 4, conv
        model.add(Convolution2D(96, 3, 3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Flatten
        model.add(Flatten())
        # layer 5, fc
        model.add(Dense(1164))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # layer 6, fc
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # layer 7, fc
        model.add(Dense(50))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        # layer 8, fc
        model.add(Dense(10))
        model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        # layer output
        model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    model.summary()
    return model

####################################################################
idx_best = 0
val_best = 9999

def main():
    nb_epoch = 5
    if len(sys.argv) >1:
        nb_epoch = int(sys.argv[1])

    print('nb_epoch = ', nb_epoch)
    print('version: balance left and right turn data')
    print('version: updated cropping and warping')
    print('version: add recovery images - this helped')
    print('version: only load centern images for recovery images - this helped')
    print('version: chagne correction factor from .25 to .2 - this helped a lot !!!!')
    print('version: try nvidia_v2 - not helped too much. need further verification')
    print('version: added back left and right images from recovery drive')
    print('version: disable upsampling')
    print('version: disable warping')
    print('version: tested with no left/right images for udacity data. removed recovery data - did not help')

    start_time = time.time()
    data_list_df = pd.read_pickle('data.pickle')

    X_train, X_val, Y_train, Y_val = load_training_and_validation_from_dataframe(data_list_df) 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'
    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model(model_type='nvidia')

    history = model.fit_generator(
        transform_generator(X_train, Y_train, batch_size = batch_size),
        samples_per_epoch=(len(X_train)),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val, batch_size = batch_size, is_validation=True),
        nb_val_samples=len(X_val),
        verbose=1)

    model.save('model.h5')

    val_loss = history.history['val_loss'][0]

    # If found a small val_loss then that's the new val_best
    if val_loss < val_best:
        val_best = val_loss
        idx_best = idx

    print("Best model found at idx:", idx_best)
    print("Best Validation score:", val_best)

    print("--- execution time %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
