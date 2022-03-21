# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/21 19:58
@Auth ： Zhe Tang
@File ：keras_mnist_v2.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)

"""
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
np.random.seed(1337)
batch_size = 128
nb_classes = 10
nb_epoch = 12
# dimensions of the image
img_rows, img_cols = 28, 28
#The number of convolution kernels.
nb_filters = 32
# Size of pooling layer.
pool_size = (2,2)
# Size of the convolution kernel.
kernel_size = (3,3)
#Load the mnist data set.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The first dimension is the sample dimension, which represents the number of samples,
# The second and third dimensions are height and width,
# The last dimension is the channel dimension, which represents the number of color channels
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#  normalized
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one-hot coding
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()

# Convolution layer, the sliding window convolution of the two-dimensional input
model.add(Convolution2D(nb_filters, kernel_size[0] ,kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# pooling layer
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# training
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# error
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
