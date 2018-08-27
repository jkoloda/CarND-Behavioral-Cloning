''' This file contains the implementation of a DNN architecture as described in
"End to End Learning for Self-Driving Cars" (Nvidia, 2016) by M. Bojarski et.al.
'''

import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Lambda
from keras.layers import Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

def get_nvidianet():
    model = Sequential()

    # Cropping
    ch, row, col = 3, 160, 320
    model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(row, col, ch)))

    # It seems more natural to normalise AFTER corpping, even though in the
    # related classroom videos the normalisation is performed before
    model.add(Lambda(lambda x: x / 127. - 1.))

    # The original NVIDIA paper works with YUV channels but we will stick to RGB

    # First 3 layers using RELUs (they are not specified in the paper though)
    # with 5x5 kernels
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))

    # Followed by 2 layers of 3x3 kernels
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    """ NOTE: From the paper: "The convolutional layers were designed to perform
    feature extraction and were chosen empirically through a series of experiments
    that varied layer configurations."

    This is pure engineering, not science (regardless what the NVIDIA guys say).
    """

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))

    # Output
    model.add(Dense(1))
    return model
