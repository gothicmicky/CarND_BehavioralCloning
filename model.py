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
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.utils import np_utils
import time

_index_in_epoch = 0
nb_epoch = 1
batch_size = 128

img_height, img_width = 160, 320

media_dir = '/media/'
def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)

def train_test_split(X, Y):
    count = int(len(X)*.8)

    X_train = X[:count]
    Y_train = Y[:count]

    X_val = X[count:]
    Y_val = Y[count:]

    return (X_train, Y_train, X_val, Y_val)

def load_training_and_validation():
    rows, labels = [], []
    correction = 0.2 # this is a parameter to tune

    with open('data/driving_log.csv', 'r') as _f:
        reader = csv.reader(_f, delimiter=',')
        next(reader, None)
        for row in reader:
            steering = float(row[3])
            throttle = float(row[4])
            brake = float(row[5])
            speed = float(row[6])

            # center camera
            rows.append(row[0].strip())
            labels.append(steering)
            
            # left camera
            rows.append(row[1].strip())
            labels.append(steering + correction)
            
            # right camera 
            rows.append(row[2].strip())
            labels.append(steering - correction)

    assert len(rows) == len(labels), 'unbalanced data'

    # shuffle the data
    X, Y = shuffle(np.array(rows), np.array(labels))

    # split into training and validation
    return train_test_split(X, Y)

def resize_image(img):
   return cv2.resize(img,(img_height, img_width))  

def augment_brightness(image):
    """
    apply random brightness on the image
    """
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    
    # scaling up or down the V channel of HSV
    image[:,:,2] = image[:,:,2]*random_bright
    return image

def affine_transform(img, angle, pixels, angle_adjust, right=True):

    cols, rows, ch = img.shape
        
    pts1 = np.float32([[10,10], [200,50], [50,250]])
    rand = np.random.uniform()

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

        # flipping Images 
        # to add additional angles
        for i in range(len(_images)):
            img = cv2.imread('{}/data/{}'.format(current, _images[i]), 1)

            if img is None: continue
            else: bad.append('/home/sameh/short-p/data/{}'.format(_images[i]))
            img = resize_image(img)
            img = img.reshape(img_height, img_width, 3)
            angle = _labels[i]
            images.append(img)
            labels.append(angle)
            # if(i == 1):
            #     cv2.imwrite('org.png', img)

            if is_validation: continue
            # Data Augmentation: 
            #flipping Images And Steering Measurements
            # affine transform (right and left)
            # To help with the left turn bias involves flipping images and 
            # taking the opposite sign of the steering measurement
            img_flipped = cv2.flip(img,1)
            images.append(img_flipped)
            labels.append(angle*-1.0)
            # if(i == 1):
            #     cv2.imwrite('flipped.png', img_flipped)
            adjust = .01
            pixels = 15
            #img, angle = affine_transform(img, labels[i], pixels, pixels*adjust*2, right=False)
            img_flipped = augment_brightness(img_flipped)
            # if(i == 1):
            #     cv2.imwrite('affined.png', img)
            images.append(img)
            labels.append(angle)

        X = np.array(images, dtype=np.float64).reshape((-1, img_height, img_width, 3))
        Y = np.array(labels, dtype=np.float64)    
        
        # raise RuntimeError(bad)
        yield (X, Y)

def gen_model(model_type = "nvidia"):
    model = Sequential()
    #data preprocessing: normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(img_height,img_width,3)))
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
        keep_prob = 0.1
        #model.add(Cropping2D(cropping=((65,25),(1,1))), input_shape=(img_height,img_width,3))
        #model.add(Lambda(lambda x: K.resize_images(x, height_factor = .5, width_factor = .125, dim_ordering = 'tf'), output_shape = (35, 40, 3)))
        model.add(Convolution2D(24,5,5,subsample=(1,1),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(36,5,5,subsample=(1,1),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(48,5,5,subsample=(1,1),activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(64,3,3,activation='relu'))
        Dropout(keep_prob)
        model.add(Convolution2D(64,3,3,activation='relu'))
        Dropout(keep_prob)
        model.add(Flatten())
        model.add(Dense(100))
        Dropout(0.5)
        model.add(Dense(50))
        Dropout(0.5)
        model.add(Dense(10))
        Dropout(0.2)
        model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def main():

    start_time = time.time()
    X_train, Y_train, X_val, Y_val = load_training_and_validation() 

    assert len(X_train) == len(Y_train), 'unbalanced training data'
    assert len(X_val) == len(Y_val), 'unbalanced validation data'
    print(len(X_train), "training images and ", len(X_val), "validation images")

    model = gen_model(model_type='nvidia')

    filepath = "weights-improvement-{epoch:02d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
        
    history_object = model.fit_generator(
        transform_generator(X_train, Y_train),
        samples_per_epoch=(len(X_train)*2),
        nb_epoch=nb_epoch,
        validation_data=transform_generator(X_val, Y_val, is_validation=True),
        nb_val_samples=len(X_val),
        verbose=1,
        callbacks=[checkpoint])

    ### print the keys contained in the history object
    ### plot the training and validation loss for each epoch
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.show()                   
    # print("Saving model weights and configuration file.")

    model.save('model.h5')
    print("--- execution time %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()
