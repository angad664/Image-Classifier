#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:07:35 2019

@author: angadsingh
"""

# part 1 building CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the CNN
classifier = Sequential()

# Step 1 Convolution     # 32 feature dector with 3 by 3 matrix
classifier.add(Conv2D(32,(3, 3), input_shape =(64, 64, 3), activation = 'relu'))

#step 2 pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding the second convolutional layer
# we only use input_shape when we dont have images before
classifier.add(Conv2D(32,(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#step 3 flattening
classifier.add(Flatten())

# step4 full connection    output_dim/units = 128 it is common practice not hard and fast rule
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) #for fully connected layer

# compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 # we have binary output so loss function is binary
# if it has more than 2 category like cats, dogs and ducks we have to use crossentropy


# part 2 fit CNN to images
#we have to do augumentation because we will get better accuracy with training set but low acc with test set
 # image augumentation prevents overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/test_set',
                                                  target_size=(64, 64), # we chose input_shape which is 64
                                                  batch_size=32,
                                                  class_mode='binary')


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,  # since we have 8000 images in out training set
                         epochs=25,    # to train the data
                         validation_data=test_set,
                         validation_steps=2000)  # that corresponds number of images in our test sets


#making new prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image) # after this run .predict (test_image) it will give error
# and ask for 4 dimension but image is only 3 d
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices  # tells which class values is specified to which here, dogs = 1
if result [0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# step 5 tuning the parameter
from wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential(optimizer)
    classifier.add(Conv2D(32,(3, 3), input_shape =(64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32,(3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
