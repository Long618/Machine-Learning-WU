# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:50:55 2019

@author: Q

利用了tensorflow搭建的神经网络框架
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import train_test_split

path='E:\Coursera-ML-using-matlab-python-master\ex4data1.mat'
data=loadmat(path)
x_data=data['X']
y=data['y']
print(x_data.shape,y.shape)
index=np.random.choice(range(5000),100)
images=x_data[index,:]
flg,ax=plt.subplots(10,10,sharex=True,sharey=True,figsize=(8,8))
for i in range(10):
    for each in range(10):
        ax[i,each].matshow(images[i*10+each].reshape(20,20),cmap='gray_r')
plt.xticks(())
plt.yticks(())
plt.show()

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
y_data=encoder.fit_transform(y.reshape(-1,1))

x=tf.placeholder(tf.float32,[None,400])
y=tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
W_L1 = tf.Variable(tf.truncated_normal([400,200],stddev=0.1))
b_L1 = tf.Variable(tf.zeros([200]))
L1 = tf.nn.sigmoid(tf.matmul(x,W_L1) + b_L1)

W_L2 = tf.Variable(tf.truncated_normal([200,50],stddev=0.1))
b_L2 = tf.Variable(tf.zeros([50]))
L2 = tf.nn.sigmoid(tf.matmul(L1,W_L2) + b_L2)

W_L3 = tf.Variable(tf.truncated_normal([50,10],stddev=0.1))
b_L3 = tf.Variable(tf.zeros([10]))
prediction =tf.matmul(L2,W_L3) + b_L3

#定义二次代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#梯度下降法
train_step=tf.train.AdamOptimizer(0.001).minimize(loss)

init=tf.global_variables_initializer()

#结果存在一个bool型列表中
#argmax返回一维张量中最大值所在位置
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率 将bool型转化为浮点型
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.2)
print(x_train.shape,x_test.shape)
with tf.Session() as sess:
    sess.run(init)
    for each in range(1001):
        sess.run(train_step,feed_dict={x:x_train,y:y_train})
        acc=sess.run(accuracy,feed_dict={x:x_test,y:y_test})
        if each%100==0:
            print('Iter'+str(each)+',testing accuracy'+str(acc))