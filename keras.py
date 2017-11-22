#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:07:53 2017

@author: wwangbt
"""
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
# Create the base pre-trained model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd 
import os as os 
import shutil as shutil


def move_image():
    df = pd.read_csv('./data/val.txt', delim_whitespace=True)
    for files in df.iloc[:,0].values:
        src = './data/'+files
        dest = './data/val/'+files
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        shutil.move(src, dest)

'''
move_image()
'''

batch_size = 32  
dim = 299 

base = InceptionV3(weights='imagenet', include_top=False)
x = base.output
x=Conv2D(128, (3, 3), padding='same')(x)
x=Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)


model = Model(input=base.input, output=predictions)
model = Model(inputs=base.input, outputs=predictions)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='constant')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/flower_photos',  
    target_size=(dim, dim),  # all images will be resized to dim * dim
    batch_size=batch_size,
    class_mode='categorical')  # since loss = categorical_crossentropy, our class labels should be categorical


test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(dim, dim),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'data/val/flower_photos',
    target_size=(dim, dim),
    batch_size=batch_size,
    class_mode='categorical')



for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True


model.compile(optimizer=optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=2569 // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=550 // batch_size)

model.evaluate_generator(validation_generator)


results = model.predict_generator(test_generator)
print(results)
X_series = pd.Series(results)
X_series.to_csv('/results.csv',encoding = 'utf-8')

file=open("/project2_20476516.txt","w")
for i in results:
    file.write(str(i)+'\n')
file.close()
