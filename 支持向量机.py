# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:52:08 2019

线性核函数
@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex6data1')
x=data['X']
y=data['y']


def plotdata(x,y):
    plt.figure(figsize=(8,5))
    plt.scatter(x[:,0],x[:,1],c=y.flatten(),cmap='rainbow')#不同y值颜色不同
    #plt.scatter(np.array(x)[:,0],np.array(x)[:,1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    
plotdata(x,y)

def plotboundary(clf,x):
    x_min,x_max=x[:,0].min()*1.2, x[:,0].max()*1.1
    y_min,y_max=x[:,1].min()*1.1, x[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)


models=[svm.SVC(C,kernel='linear') for C in [1,100]]
clfs=[model.fit(x,y.ravel())for model in models]

titles=['svm decision boundary with C ={}(dataset1)'.format(C) for C in [1,100]]
for model,title in zip(clfs,titles):
    plt.figure(figsize=(8,5))
    plotdata(x,y)
    plotboundary(model,x)
    plt.title(title)
