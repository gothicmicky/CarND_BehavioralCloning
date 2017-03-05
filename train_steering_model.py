import os
import pickle
import json
import random
import csv

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils

_index_in_epoch = 0
nb_epoch = 10
batch_size = 128

img_height, img_width = 64, 64

def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

def train_test_split(X, Y):
    count = int(len(X)*.7)

    X_train = X[:count]
    Y_train = Y[:count]

    X_val = X[count:]
    Y_val = Y[count:]

    return (X_train, Y_train, X_val, Y_val)

def load_training_and_validation():
    rows, labels = [], []
    with open('data/driving_log.csv', 'r') as _f:
        reader = csv.reader(_f, delimiter=',')
        next(reader, None)
        for row in reader:
            rows.append(row[0].strip())
            labels.append(float(row[3]))
            # left camera
            rows.append(row[1].strip())
            labels.append(float(row[3])+.25)
            # right camera 
            rows.append(row[2].strip())
            labels.append(float(row[3])-0.25)

    assert len(rows) == len(labels), 'unbalanced data'

    # shuffle the data
    X, Y = shuffle(np.array(rows), np.array(labels))

    # split into training and validation
    return train_test_split(X, Y)

def resize_image(img):
   return cv2.resize(img,( 64, 64))  

def affine_transform(img, angle, pixels, angle_adjust, right=True):

    cols, rows, ch = img.shape
        
    pts1 = np.float32([[10,10], [200,50], [50,250]])

    if right:
        pts2 = np.float32([[10, 10], [200+pixels, 50], [50, 250]])
        angle =- angle_adjust
    else:
        pts2 = np.float32([[10, 10], [200-pixels, 50], [50, 250]])
        angle =- angle_adjust

    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (rows, cols))

    return dst.reshape((cols, rows, ch)), angle


def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
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


def transform_generator(x, y, batch_size=32, is_validation=False):
    while True:
        bad = []
        images, labels = list(), list()

        _images, _labels = next_batch(x, y, batch_size)

        pixels = 15
        adjust = .01

        current = os.path.dirname(os.path.realpath(__file__))

        # read in images as grayscale
        # affine transform (right and left)
        # to add additional angles
        for i in range(len(_images)):
            img = cv2.imread('{}/data/{}'.format(current, _images[i]), 1)

            if img is None: continue
            else: bad.append('/home/sameh/short-p/data/{}'.format(_images[i]))

            img = resize_image(img)

            img = img.reshape(img_height, img_width, 3)
            images.append(img)
            labels.append(_labels[i])

            if is_validation: continue

            img, angle = affine_transform(img, labels[i], pixels, pixels*adjust*2, right=True)
            images.append(img)
            labels.append(angle)

        X = np.array(images, dtype=np.float64).reshape((-1, img_height, img_width, 3))

        X /= 255.

        Y = np.array(labels, dtype=np.float64)

        # raise RuntimeError(bad)

        yield (X, Y)

def gen_model():
    model = Sequential()

    # (((64 - 5) + 4) / 1.) + 1
    model.add(Convolution2D(16, 5, 5, subsample=(1, 1), input_shape=(img_height, img_width, 3)))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((64 - 5) + 4) / 1.) + 1
    model.add(Convolution2D(32, 5, 5, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (((64 - 5) + 4) / 1.) + 1
    model.add(Convolution2D(64, 5, 5, subsample=(1, 1)))
    model.add(ZeroPadding2D(padding=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)

    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def main():

    X_train, Y_train, X_val, Y_val = load_training_and_validation() 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'

    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model()

    filepath = "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
        
    model.fit_generator(
        transform_generator(X_train, Y_train),
        samples_per_epoch=(len(X_train)*2),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val, is_validation=True),
        nb_val_samples=len(X_val),
        callbacks=[checkpoint])

    print("Saving model weights and configuration file.")

    if not os.path.exists("./outputs/sim"):
        os.makedirs("./outputs/sim")

    model.save('model.h5')

if __name__ == '__main__':
    main()
