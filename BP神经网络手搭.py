# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:32:13 2019

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report 

path='E:\Coursera-ML-using-matlab-python-master\ex4data1.mat'
data=loadmat(path)
x=data['X']
y=data['y']


index=np.random.choice(range(5000),100)
images=x[index,:]
flg,ax=plt.subplots(10,10,sharex=True,sharey=True,figsize=(8,8))
for i in range(10):
    for each in range(10):
        ax[i,each].matshow(images[i*10+each].reshape(20,20),cmap='gray_r')
plt.xticks(())
plt.yticks(())
plt.show()

#对y进行onehot编码
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
y_data=encoder.fit_transform(y.reshape(-1,1))
x_data=np.insert(x, 0, 1, axis=1)
#(5000, 401) (5000, 10)

#获取已经训练好点的权重
def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2'] 

t1, t2 = load_weight('E:\Coursera-ML-using-matlab-python-master\ex4weights.mat')
#(25,401),(10,26)

#需要将参数矩阵展开，才能传入优化函数
def serialize(a,b):
    return np.r_[a.flatten(),b.flatten()] #np.r_对于数组而言存放在同一行里，b在a后面
#对于矩阵 np.r_[]则是按列添加矩阵，b在a下面

theta = serialize(t1, t2)
#theta.shape (10285,)

#提取参数，将参数形式转变回来
def deserialize(seq):
    return seq[:25*401].reshape(25,401),seq[25*401:].reshape(10,26)

def sigmoid(z):
    return 1/(1+np.exp(-z))

#前馈网络
def feed_forward(theta,x,):
    t1, t2 = deserialize(theta)
    # 前面已经插入过偏置单元，这里就不用插入了
    a1 = x
    z2 = a1.dot(t1.T)
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)
    z3 = a2.dot(t2.T)
    h = sigmoid(z3)
    return a1, z2, a2, z3, h 

#定义代价函数
def cost(theta,x,y):
    a1,z2,a2,z3,h=feed_forward(theta,x_data)
    J=0
    for i in range(len(x)):
        first=-y[i]*np.log(h[i])
        second=(1-y[i])*np.log(1-h[i])
        J+=np.sum(first-second)
    J=J/len(x)
    return J

#定义正则化代价函数
def regularized_cost(theta,x,y,l=1):
    t1,t2=deserialize(theta)
    reg=np.sum(t1[1:,1:]**2)+np.sum(t2[1:,1:]**2)
    return 1/(2+len(x))*reg+cost(theta,x,y)

print(regularized_cost(theta,x_data,y_data,1))

#后向传播
def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))

def random_init(size):
    return np.random.uniform(-0.12,0.12,size)#均匀分布区间

#对参数求梯度
def gradient(theta,x,y):
    t1,t2=deserialize(theta)
    a1,z2,a2,z3,h=feed_forward(theta,x)
    d3=h-y
    d2=d3.dot(t2[:,1:])*sigmoid_gradient(z2)#z2为（5000，25），t2为（10,26）
    D2=(d3.T).dot(a2)
    D1=(d2.T).dot(a1)
    D=(1/len(x))*serialize(D1,D2)
    return D
#a1 (5000, 401) t1 (25, 401)
#z2 (5000, 25)
#a2 (5000, 26) t2 (10, 26)
#z3 (5000, 10)
#a3 (5000, 10)

#正则化神经网络
def regularized_gradient(theta,x,y,l=1):
    a1,z2,a2,z3,h=feed_forward(theta,x)
    D1,D2=deserialize(gradient(theta,x,y))#求出t1,t2的梯度D1,D2
    t1[:,0]=0
    t2[:,0]=0#对偏置单元的不进行惩罚
    reg_D1=D1+(l/len(x))*t1
    reg_D2=D2+(l/len(x))*t2
    return serialize(reg_D1,reg_D2)

#优化参数
    #初始化参数theta
def nn_training(x, y):
    init_theta = random_init(10285)  # 初始化参数

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(x, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res

res=nn_training(x_data,y_data)

def accuracy(theta,x,y):
    _,_,_,_,h=feed_forward(theta,x)
    y_pred=np.argmax(h,axis=1)+1#1所在的索引+1 即为数字几
    acc=0
    for i in range(5000):
        if y[i]==y_pred[i]:
            acc+=1
    print('accuracy is',acc/len(y))
    print(classification_report(y, y_pred))

accuracy(res.x,x_data,y)

#计算出来准确率和直接tensorflow里面套模型差不多