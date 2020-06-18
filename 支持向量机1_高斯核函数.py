# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:23:19 2019

高斯核函数

@author: Q
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

data=data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex6data2.mat')
x=data['X']
y=data['y']

from sklearn import cross_validation
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size = 0.5)

def plotdata(x,y):
    plt.figure(figsize=(8,5))
    plt.scatter(x[:,0],x[:,1],c=y.flatten(),cmap='rainbow')#不同y值颜色不同
    #plt.scatter(np.array(x)[:,0],np.array(x)[:,1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    

def plotboundary(clf,x):
    x_min,x_max=x[:,0].min()*1.2, x[:,0].max()*1.1
    y_min,y_max=x[:,1].min()*1.1, x[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)


def gaussKernel(x1,x2,sigma):
    return np.exp(- ((x1 - x2) ** 2).sum() / (2 * sigma ** 2))

sigma=0.1
gamma=np.power(sigma,-2.)
clf=svm.SVC(C=1,kernel='rbf',gamma=gamma)
model=clf.fit(x_train,y_train.flatten())
plotdata(x_train,y_train)
plotboundary(model,x_train)
print(clf.score(x_train,y_train))
print(clf.score(x_test,y_test))

#print(model.score(x_train,y_train))           0.9930394431554525
#print(model.score(x_test,y_test))             0.9884259259259259
'''
y_pred1=model.predict(x_train)
y_pred2=model.predict(x_test)
from sklearn.metrics import classification_report 
print('train ',classification_report(y_train,y_pred1))
print('test ',classification_report(y_test,y_pred2))
plotdata(x_test,y_test)
plotboundary(model,x_test)
'''