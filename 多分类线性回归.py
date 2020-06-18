# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 17:12:01 2019

@author: Q
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex3data1.mat')
X=data['X']
y=data['y']
#数据可视化
num=np.random.randint(0,5000)
image=X[num,:]
fig,ax=plt.subplots(figsize=(1,1))
ax.matshow(image.reshape((20,20)),cmap='gray_r')
plt.show()

index=np.random.randint(0,X.shape[0],100)
image=X[index,:]
fig,ax=plt.subplots(nrows=10, ncols=10,sharey=True, sharex=True,figsize=(8, 8))
for i in range(10):
    for each in range(10):
        ax[i,each].matshow(image[i*10+each].reshape(20,20),cmap='gray_r')
plt.xticks([])
plt.yticks([])
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def regularized_cost(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        X: feature matrix, (m, n+1) # 插入了x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
    """
    thetaReg = theta[1:]
    first = (-y*np.log(sigmoid(X.dot(theta)))) + (y-1)*np.log(1-sigmoid(X@theta))
    reg = (thetaReg@thetaReg)*l / (2*len(X))
    return np.mean(first) + reg

def regularized_gradient(theta, X, y, l):
    """
    don't penalize theta_0
    args:
        l: lambda constant
    return:
        a vector of gradient
    """
    thetaReg = theta[1:]
    first = (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
    # 这里人为插入一维0，使得对theta_0不惩罚，方便计算
    reg = np.concatenate([np.array([0]), (l / len(X)) * thetaReg])
    return first + reg


from scipy.optimize import minimize

def one_vs_all(X, y, l, K):
    """generalized logistic regression
    args:
        X: feature matrix, (m, n+1) # with incercept x0=1
        y: target vector, (m, )
        l: lambda constant for regularization
        K: numbel of labels
    return: trained parameters
    """
    all_theta = np.zeros((K, X.shape[1]))  # (10, 401)

    for i in range(1, K+1):
        theta = np.zeros(X.shape[1])
        y_i = np.array([1 if label == i else 0 for label in y])

        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                        jac=regularized_gradient, options={'disp': True})
        all_theta[i-1,:] = ret.x

    return all_theta


def predict_all(X, all_theta):
    # compute the class probability for each class on each training instance
    h = sigmoid(X @ all_theta.T)  # 注意的这里的all_theta需要转置
    # create array of the index with the maximum probability
    # Returns the indices of the maximum values along an axis.
    h_argmax = np.argmax(h, axis=1)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax
#这里的h共5000行，10列，每行代表一个样本，每列是预测对应数字的概率。我们取概率最大对应的index加1就是我们分类器最终预测出来的类别。返回的h_argmax是一个array，包含5000个样本对应的预测值。

raw_X, raw_y = X,y
X = np.insert(raw_X, 0, 1, axis=1) # (5000, 401)
y = raw_y.flatten()  # 这里消除了一个维度，方便后面的计算 or .reshape(-1) （5000，）

all_theta = one_vs_all(X, y, 1, 10)

y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred == y)
print ('accuracy = {0}%'.format(accuracy * 100))