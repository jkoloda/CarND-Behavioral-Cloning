'''
This script reads the data, implements a data generator with online data
augmentation and performs the actual training process.
'''

import os
import csv
import cv2
import keras
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from model import get_nvidianet

path = os.getcwd()

# ----------------------- DATA EXTRACTION --------------------------------------
# Gather info for samples provided within this project
samples = []
prefix = os.path.join(path, 'data/example/')
with open(os.path.join(prefix, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
# Delete the first line containing labels
del samples[0]
# Completing image filenames by full path
for ii in range(0, len(samples)):
    samples[ii][0] = os.path.join(prefix, samples[ii][0])
    samples[ii][1] = os.path.join(prefix, samples[ii][1])
    samples[ii][2] = os.path.join(prefix, samples[ii][2])

# The same process is repeated for samples gathered
# by running the simulator by ourselves
mysamples = []
prefix = os.path.join(path, 'data/data/')
with open(os.path.join(prefix, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        mysamples.append(line)
del mysamples[0]
for ii in range(0, len(mysamples)):
    mysamples[ii][0] = os.path.join(prefix, mysamples[ii][0])
    mysamples[ii][1] = os.path.join(prefix, mysamples[ii][1])
    mysamples[ii][2] = os.path.join(prefix, mysamples[ii][2])

# Put all data together
samples = samples + mysamples

# Perform train-validation splitting (80% - 20%)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# ------------------------------------------------------------------------------


# ------------------------- DATA GENERATOR -------------------------------------
# Generator for data acquisition
def generator(samples, batch_size=32, augmentation=True):
    '''
    Generates batches of data with optional online data augmentation
    consisting on horizontally mirroring an image (with a probability of 50%).

    Parameters
    ----------
    samples : list
        list containing infirmation for each image sample

    batch_size : int
        number of samples per batch

    augmentation : bool
        indicates whether data augmentation is to be applied
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates

        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                if augmentation == True:
                    if np.random.rand() > 0.5:
                        images.append(center_image)
                        angles.append(center_angle)
                    else:
                        images.append(np.fliplr(center_image))
                        angles.append(-center_angle)
                else:
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Build data generators for training and validation
train_generator = generator(train_samples, batch_size=32, augmentation=True)
validation_generator = generator(validation_samples, batch_size=32, augmentation=False)
# ------------------------------------------------------------------------------


# ------------------------------- TRAINING -------------------------------------

checkpoint = keras.callbacks.ModelCheckpoint(
                filepath='models/model.{epoch:02d}-{val_loss:.5f}.h5',
                monitor='val_loss',
                verbose=0, save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=1)
callbacks_list = [checkpoint]

model = get_nvidianet()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                              samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=25,
                              callbacks=callbacks_list)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(history.history.keys())
print(model.summary())

# Show history
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
fig.savefig('temp.png')
