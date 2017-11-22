# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:05:28 2017

@author: 45183
"""

import tensorflow as tf
import os
import numpy as np
from skimage import io,transform

def read_index(path):
    '''
    to read the index
    '''
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
    '''compare true and false and return error rate'''    
    sum_e=0
    for i in range(len(pre_y)):
        if pre_y[i]!=true_y[i]:
            sum_e=sum_e+1
    error=sum_e/len(pre_y)
    return error

def read_x_y(path,w,h):
    '''w,h is the wid anf height of picture we use next step'''
    x_index,y=read_index(path)
    x=[]
#    y=y[0:10]
    for i in range(len(y)):
        pic_raw=io.imread(x_index[i])
        ##use k-nearst way to protect the picture
        #pic=tf.image.resize_images(pic_raw,(w,h),method=1)
        pic=transform.resize(pic_raw,(w,h))
        x.append(pic)
        print(i)
    return x,y

''' 
import matplotlib.pyplot as plt  
with tf.Session() as sess:
    pic_raw=io.imread(x_index[i])
    pic=tf.image.resize_images(pic_raw,(100,100),method=0)
    plt.imshow(pic.eval())

'''
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
 





path='train'    
w=h=100
c=3   
    
x,y=read_x_y(path,w,h)
y=np.asarray(y,np.int32)
x=np.asarray(x,np.float32)

path='val'      
x_val,y_val=read_x_y(path,w,h)
y_val=np.asarray(y_val,np.int32)
x_val=np.asarray(x_val,np.float32)



##constrain network
x_place=tf.placeholder(tf.float32,shape=[None,w,h,c])
y_place=tf.placeholder(tf.int32,shape=[None,])

#first one 
conv1=tf.layers.conv2d(
      inputs=x_place,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#100-50
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#next one
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#25
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#12
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#6
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

dense1 = tf.layers.dense(inputs=re1, 
                      units=1024, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2= tf.layers.dense(inputs=dense1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2, 
                        units=5, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_place,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
predict=tf.cast(tf.argmax(logits,1),tf.int32)
correct_prediction = tf.equal(predict, y_place)    
acc= tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


num_epoch=30

batch_size=64
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
for epoch in range(num_epoch):
    print("epoch",epoch)
    #training
    train_loss, train_acc = 0, 0
    for x_train_a, y_train_a in minibatches(x, y, batch_size, shuffle=True):
        sess.run(train_op, feed_dict={x_place: x_train_a, y_place: y_train_a})
        loss_local=sess.run(loss, feed_dict={x_place: x_train_a, y_place: y_train_a})
        acc_local=sess.run(acc, feed_dict={x_place: x_train_a, y_place: y_train_a})
        train_loss += loss_local; train_acc += acc_local
    print("   train loss: %f" % (train_loss/ len(y)))
    print("   train acc: %f" % (train_acc/ len(y)))
    
    #validation
    val_loss, val_acc = 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        loss_local=sess.run(loss, feed_dict={x_place: x_val_a, y_place: y_val_a})
        acc_local=sess.run(acc, feed_dict={x_place: x_val_a, y_place: y_val_a})
        val_loss += loss_local; val_acc += acc_local
    print("   validation loss: %f" % (val_loss/ len(y_val)))
    print("   validation acc: %f" % (val_acc/ len(y_val)))
    #lost any thing?
    predict_val=sess.run(predict, feed_dict={x_place: x_val})
    error_rate_val=error_rate(predict_val,y_val)
    print(error_rate_val+val_acc/ len(y_val))
    
    filename="model"+str(epoch)
    os.mkdir(filename)
    saver=tf.train.Saver()
    path=filename+"/model.ckpt"
    saver.save(sess,path)
sess.close()

'''
saver=tf.train.Saver()
epoch=21
filename="model"+str(epoch)
path=filename+"/model.ckpt"
sess=tf.InteractiveSession() 
#saver.restore(sess,path)
predict_val=sess.run(predict, feed_dict={x_place: x_val})
error_rate_val=error_rate(predict_val,y_val)
print(1-error_rate_val)
val_loss, val_acc = 0, 0
for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
    loss_local=sess.run(loss, feed_dict={x_place: x_val_a, y_place: y_val_a})
    acc_local=sess.run(acc, feed_dict={x_place: x_val_a, y_place: y_val_a})
    val_loss += loss_local; val_acc += acc_local
print("   validation loss: %f" % (val_loss/ len(y_val)))
print("   validation acc: %f" % (val_acc/ len(y_val)))
#lost any thing?
predict_val=sess.run(predict, feed_dict={x_place: x_val})
error_rate_val=error_rate(predict_val,y_val)
print(error_rate_val+val_acc/ len(y_val))

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
for i in range(len(x_index)):
    pic_raw=io.imread(x_index[i])
    pic=transform.resize(pic_raw,(w,h))
    x_test.append(pic)
    
predict_test=sess.run(predict, feed_dict={x_place: x_test})
file=open("project2_20476516.txt","w")
for i in predict_test:
    file.write(str(i)+'\n')
file.close
sess.close()
'''
