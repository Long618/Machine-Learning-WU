# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:29:27 2019

@author: Q
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#读取数据
path =  'E:\Coursera-ML-using-matlab-python-master\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])  
"""
#可看出样本的平均值，最大值，方差等
print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()
#数据作图
"""
'''
def computeCost(X, y, theta):
    inner = np.power(((X * theta) -  y), 2)
    return np.sum(inner) / (2 * len(X))
'''
#data.insert(0,'ones',1)#在第0列插入名为ones值的一列1
X = data.iloc[:,0:1]  # 取前cols-1列，即输入向量
y = data.iloc[:,1:2] # 取最后一列，即目标向量
X = np.mat(X.values)
y = np.mat(y.values)
b=tf.Variable(0.)
k=tf.Variable(0.)
y_data=k*X+b
with tf.name_scope('loss'):
#定义二次代价函数
    loss=tf.reduce_mean(tf.square(y_data-y))
    tf.summary.scalar('loss',loss)

merged=tf.summary.merge_all()

train=tf.train.GradientDescentOptimizer(0.005).minimize(loss)

init=tf.global_variables_initializer()

cost = np.zeros(1001)
with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for step in range(1001):
        summary, _=sess.run([merged,train])
        writer.add_summary(summary,step)
    
    prediction_value=sess.run(y_data)
    plt.figure()
    plt.scatter(X.tolist(),y.tolist())
    plt.plot(X,prediction_value,'r-',lw=5)
    plt.show()   