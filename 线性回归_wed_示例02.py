# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:29:27 2019

示例2
@author: Q
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

#读取数据
path =  'E:\Coursera-ML-using-matlab-python-master\ex1data2.txt'
data = pd.read_csv(path,names=['Size', 'Bedrooms', 'Price'])  

data = (data - data.mean()) / data.std()#对数据进行归一化

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0,'ones',1)#在第0列插入名为ones值的一列1
clos=data.shape[1]#data有多少列
X = data.iloc[:,:clos-1]  # 取前cols-1列，即输入向量
y = data.iloc[:,clos-1:clos] # 取最后一列，即目标向量
X = np.mat(X.values)
y = np.mat(y.values)#将X，y转换成numpy矩阵
theta = np.mat([0,0,0])
#写一个梯度下降器

def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""
    #epoch是迭代次数
    #parameters = int(theta.flatten().shape[1])  # flatten（）默认按行降维。然后计算参数θ 有多少个
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # X有多少行，即样本数量m
    
    for i in range(epoch):
        # 利用向量化一步求解
        temp =theta - (alpha / m) * (X * theta.T - y).T * X #更新参数         
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

with tf.Session() as sess:
    #当前目录logs文件夹写入这个文件，如果没有则创建一个。存放graph这个图结构
    write=tf.summary.FileWriter('logs/',sess.graph)

#定义学习率和迭代次数
alpha = 0.01
epoch = 1000
#调用梯度下降器
final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()