#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:07:53 2017

@author: wwangbt
"""
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd 
import os as os 
import shutil as shutil
from keras.preprocessing import image
import numpy as np

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
#model = Model(inputs=base.input, outputs=predictions)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='constant')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'data/flower_photos',  
    target_size=(dim, dim),  # image size dim * dim
    batch_size=batch_size,
    class_mode='categorical') 


test_generator = test_datagen.flow_from_directory(
    'data/final',
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

path_test='test.txt'
file=open(path_test)
try:
    file_context=file.read()
finally:
    file.close()
    x_index=file_context.split('\n')
if x_index[len(x_index)-1]=='':
    x_index=x_index[0:len(x_index)-1]

def choose_class(x):
    result=[]
    for i in range(len(x)):
        result.append(list(x[i]).index(max(x[i]))
    return result
        
    


test_x=[]
for path in x_index:
    img = image.load_img(path, target_size=(dim, dim))
    x = image.img_to_array(img)
    test_x.append(x)
test_x=np.array(test_x)
test_x=test_x/255
'''
path='val'    

import tensorflow as tf
import os
import numpy as np
from skimage import io,transform

def read_index(path):

    path=path+'.txt'
    file = open(path) #
    try:
        file_context = file.read() 
        #  file_context = open(file).read().splitlines() 
    
    finally:
        file.close()
    file_context=file_context.replace('\n', ' ').split(' ')
    
    if len(file_context)%2==1:
        len_of_context=len(file_context)-1
    else:
        len_of_context=len(file_context)    
            
    
    x_index=list([])
    y=list([])
    for i in range(len_of_context):
        if i%2==0:
            x_index.append(file_context[i])
        else:
            y.append(file_context[i])
    return x_index,y
 
def error_rate(pre_y,true_y):    
  
    sum_e=0
    for i in range(len(pre_y)):
        if pre_y[i]!=true_y[i]:
            sum_e=sum_e+1
    error=sum_e/len(pre_y)
    return error


  
x_val_index,y_val=read_index(path)
y_val=np.asarray(y_val,np.int32)
val_x=[]
for path in x_val_index:
    img = image.load_img(path, target_size=(dim, dim))
    x = image.img_to_array(img)
    val_x.append(x)
val_x=np.array(val_x)
val_x=val_x/255
results_val_p = model.predict(val_x,verbose=1)
result_val=choose_class(results_val_p)
def error_rate(pre_y,true_y):        
    sum_e=0
    for i in range(len(pre_y)):
        if pre_y[i]!=true_y[i]:
            sum_e=sum_e+1
    error=sum_e/len(pre_y)
    return error
acc_val=1-error_rate(result_val,y_val)
print(acc_val)

'''




results = model.predict(test_generator)
print(results)
X_series = pd.Series(results)
X_series.to_csv('/results.csv',encoding = 'utf-8')

file=open("/project2_20476516.txt","w")
for i in results:
    file.write(str(i)+'\n')
file.close()
