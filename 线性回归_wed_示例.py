# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:29:27 2019

示例
@author: Q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#读取数据
path =  'E:\Coursera-ML-using-matlab-python-master\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])  
"""
#可看出样本的平均值，最大值，方差等
print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()
#数据作图，查看数据
"""

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) -  y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0,'ones',1)#在第0列插入名为ones值的一列1
clos=data.shape[1]#data有多少列
X = data.iloc[:,:clos-1]  # 取前cols-1列，即输入向量
y = data.iloc[:,clos-1:clos] # 取最后一列，即目标向量
X = np.mat(X.values)
y = np.mat(y.values)#将X，y转换成numpy矩阵
theta = np.mat([0,0])
#计算代价函数
computeCost(X, y, theta)#求代价函数的初值
#写一个梯度下降器
def gradientDescent(X, y, theta, alpha, epoch):
    """reuturn theta, cost"""
    #epoch是迭代次数
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个 θ 临时矩阵(1, 2)
    parameters = int(theta.flatten().shape[1])  # flatten（）默认按行降维。然后计算参数θ 有多少个
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0]  # X有多少行，即样本数量m
    
    for i in range(epoch):
        # 利用向量化一步求解
        temp =theta - (alpha / m) * (X * theta.T - y).T * X #更新参数         
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

#定义学习率和迭代次数
alpha = 0.01
epoch = 1000
#调用梯度下降器
final_theta, cost = gradientDescent(X, y, theta, alpha, epoch)
x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标，在x范围内均匀生成100个点
f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，用求得的theta来绘出利润
#x=data.iloc[:,clos-2:clos-1]
#f=X * final_theta.T
fig, ax = plt.subplots(figsize=(6,4))#绘出图片比例为6:4
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)  # 将上面的label放置在图的哪里，2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()