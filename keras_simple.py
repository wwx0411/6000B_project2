#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:18:18 2017

@author: wwangbt
"""

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Activation, Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd 
import os as os 
import shutil as shutil
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical

#imput data
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
def read_x_y(path,dim):
    x_index,y=read_index(path)
    y=np.asarray(y,np.int32)
    x=[]
    for path in x_index:
        img = image.load_img(path, target_size=(dim, dim))
        x_local = image.img_to_array(img)
        x.append(x_local)
    x=np.array(x)
    return x,y

def class_to_val(x):
    x_val=[]
    for i in x:
        a=[0,0,0,0,0]
        a[i]=1
        x_val.append(a)
    x=np.array(x)
    return x
    

x_train,y_train=read_x_y('train',299)
x_val,y_val=read_x_y('val',299)
x_train=x_train/255
x_val=x_val/255

y_train = to_categorical(y_train, 5)
y_val = to_categorical(y_val, 5)
print(y_train.shape)
print(y_val.shape)

path_test='test.txt'
file=open(path_test)
try:
    file_context=file.read()
finally:
    file.close()
    x_index=file_context.split('\n')
if x_index[len(x_index)-1]=='':
    x_index=x_index[0:len(x_index)-1]
x_test=[]
for path in x_index:
    pic=image.load_img(path, target_size=(299, 299))
    x_local = image.img_to_array(pic)
    x_test.append(x_local)
x_test=np.array(x_test)
x_test=x_test/255

base = InceptionV3(weights='imagenet', include_top=False,input_shape=[299,299,3])
x = base.output
'''
x=Conv2D(128, (3, 3), padding='same')(x)
x=Activation('relu')(x)
x = GlobalAveragePooling2D()(x)
'''
x=Flatten()(x)
x = Dense(1024, activation='relu')(x)
x=Dropout(0.7)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(input=base.input, output=predictions)


model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=2,validation_data=(x_val,y_val))


y_test=model.predict(x_test)
result = np.argmax(y_test, axis=1).shape


'''
file=open("/project2_20476516.txt","w")
for i in results:
    file.write(str(i)+'\n')
file.close()
path=/project2_20476516.csv"
np.savetxt(path, result, fmt=%d, delimiter=",")
'''