# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:16:58 2017

@author: 45183
"""

def minibatches(x, y, batch_size=1, shuffle=False):
    assert len(x) == len(y)
    if shuffle:
        index = np.arange(len(y))
        np.random.shuffle(index)
    for start_index in range(0, len(x) - batch_size + 1, batch_size):
        if shuffle:
            local_index = index[start_index:start_index + batch_size]
        else:
            local_index = slice(start_index, start_index + batch_size)
        yield x[local_index], y[local_index]
    if(start_index+batch_size<len(y)-1):
        if shuffle:
            local_index=index[start_index + batch_size:len(y)]
        else:
            local_index = slice(start_index + batch_size,len(y))
        yield x[local_index], y[local_index]
    


from skimage import io,transform
import glob
import os
import numpy as np
import time
x=y=np.array([1,2,3,4,5,6])
for x_train_a, y_train_a in minibatches(x, y,batch_size=4,shuffle=True):
    #print(i)
    print(y_train_a)
    i=i+1