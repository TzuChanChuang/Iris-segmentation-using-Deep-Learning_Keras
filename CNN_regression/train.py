from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Activation, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 280
img_cols = 320


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)


    conv7 = Convolution2D(2048, 3, 3, activation='relu', border_mode='same')(pool6)
    conv7 = Convolution2D(2048, 3, 3, activation='relu', border_mode='same')(conv7)


    #drop = Dropout(0.25)(conv7)
    flat = Flatten()(conv7) 
    out = Dense(6)(flat) 

    model = Model(input=inputs, output=out)
    print(model.summary())

    model.compile(optimizer=Adam(lr=1e-5), loss='mse') ##

    return model


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_train_label = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_train_label, batch_size=32, nb_epoch=200, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = load_test_data()

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting results on test data...')
    print('-'*30)
    imgs_test_result = model.predict(imgs_test, verbose=1)
    np.save('imgs_test_result.npy', imgs_test_result)
    
    test_result = np.load('imgs_test_result.npy');
    np.savetxt("test_result.csv", test_result, delimiter=",")


if __name__ == '__main__':
    train_and_predict()