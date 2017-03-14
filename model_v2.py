import os
import pickle
import json
import random
import csv
import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from preprocess import img_preprocess
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

_index_in_epoch = 0
nb_epoch = 8
batch_size = 128
img_height, img_width = 160, 320
media_dir = '/media/'

####################################################################

def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

####################################################################

def load_training_and_validation():
    rows, labels = [], []
    correction = 0.25 # this is a parameter to tune

    with open('data/driving_log.csv', 'r') as _f:
        reader = csv.reader(_f, delimiter=',')
        next(reader, None)
        for row in reader:
            steering = float(row[3])
            throttle = float(row[4])
            brake = float(row[5])
            speed = float(row[6])
            
            #cam_view = np.random.choice(['center', 'left', 'right'])
            #if cam_view == 'left':
            ## left image
            rows.append(row[1].strip())
            labels.append(steering + correction)

            #elif cam_view == 'center':
            ## centre image
            rows.append(row[0].strip())
            labels.append(steering)

            #elif cam_view == 'right':
                ## right image
            rows.append(row[2].strip())
            labels.append(steering - correction)
    
    with open('data/driving_log_recovery.csv', 'r') as _f:
        reader = csv.reader(_f, delimiter=',')
        next(reader, None)
        for row in reader:
            steering = float(row[3])
            throttle = float(row[4])
            brake = float(row[5])
            speed = float(row[6])
            
            rows.append(row[1].strip())
            labels.append(steering + correction)

            ## centre image
            rows.append(row[0].strip())
            labels.append(steering)

            ## right image
            rows.append(row[2].strip())
            labels.append(steering - correction)

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
        current = os.path.dirname(os.path.realpath(__file__))

        for i in range(len(_images)):
            img = cv2.imread('{}/data/{}'.format(current, _images[i]))

            if img is None: continue
            else: bad.append('/home/sameh/short-p/data/{}'.format(_images[i]))
            img = img.reshape(img_height, img_width, 3)
            angle = _labels[i]

            #pre-processing
            img, angle = img_preprocess(img, angle)
            img = cv2.resize(img, (200,66))

            # if is_validation: 
            #     images.append(img)
            #     labels.append(angle)
            #     continue

            #upsample and downsample:
            if (angle > 0.1):
                images.append(img)
                labels.append(angle)

                # Adding a small deviation of the angle 
                # This is to create more right turning samples for the same image
                for i in range(10):
                    new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                    images.append(img)
                    labels.append(new_angle)
            
            elif (angle < -0.1):
                images.append(img)
                labels.append(angle)
        
                # Adding a small deviation of the angle
                # This is to create more left turning samples for the same image
                for i in range(6):
                    new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
                    images.append(img)
                    labels.append(new_angle)
            
            else:
                if (angle != 0.0):
                    # Include all near 0 angle data
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

    adam = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

####################################################################

def main():

    print('version: balance left and right turn data')
    print('version: updated cropping and warping')

    start_time = time.time()
    X_train, X_val, Y_train, Y_val = load_training_and_validation() 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'
    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model(model_type='nvidia')

    #filepath = "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    #checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
 
    history = model.fit_generator(
        transform_generator(X_train, Y_train, batch_size = batch_size),
        samples_per_epoch=(len(X_train)),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val, batch_size = batch_size, is_validation=True),
        nb_val_samples=len(X_val),
        verbose=1)
        #callbacks=[checkpoint])

    model.save('model.h5')
    print("--- execution time %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
