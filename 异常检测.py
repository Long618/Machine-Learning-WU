# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:17:30 2019

@author: Q
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex8data1.mat')
x,xval,yval=data['X'],data['Xval'],data['yval']
print(x.shape,xval.shape,yval.shape)

def plotdata():
    plt.figure(figsize=(8,5))
    plt.plot(x[:,0],x[:,1],'bx')

#多变量高斯分布
def gaussian(x,mu,sigma):
    m,n=x.shape
    if np.ndim(sigma)==1:#sigma是一个一维数组
        sigma=np.diag(sigma)#sigma变成sigma为对角线的矩阵
        
    norm=1./(np.power((2*np.pi),n/2)*np.sqrt(np.linalg.det(sigma)))
    exp=np.zeros((m,1))
    for row in range(m):
        xrow=x[row]
        exp[row]=np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma)).dot(xrow-mu))
    return norm*exp

def getGaussianParams(x,useMultivariate):
    mu=x.mean(axis=0)#对每一列求平均值
    if useMultivariate:
        sigma=((x-mu).T).dot(x-mu)/len(x)#多元高斯函数的sigma
    else:
        sigma=x.var(axis=0,ddof=0)#var为方差，std为标准差
    return mu,sigma

def plotContours(mu,sigma):
    delta=.3
    x=np.arange(0,30,delta)
    y=np.arange(0,30,delta)
    
    xx,yy=np.meshgrid(x,y)
    points=np.c_[xx.ravel(),yy.ravel()]#一列是xx,一列是yy。一列是横坐标，一列是纵坐标
    z=gaussian(points,mu,sigma).reshape(xx.shape)
    
    cont_levels=[10**h for h in range(-20,0,3)]#给出的，可以通过求解的概率推倒出来
    plt.contour(xx,yy,z,cont_levels)
    
    plt.title('gaussian contours')

'''
plotdata()
useMV=False
plotContours(*getGaussianParams(x,useMV))

useMV=True
plotdata()
plotContours(*getGaussianParams(x,useMV))
从上面的图可以看到，一元高斯模型仅在横向和纵向上有变化，
而多元高斯模型在斜轴上也有相关变化，对应着特征间的相关关系。
而一元高斯模型就是多元高斯模型中协方差矩阵为对角矩阵的结果，
即协方差都为0，不考虑协方差，只考虑方差，故一元高斯模型不会有斜轴上的变化。
从上面的图我们可以清晰的看到，哪些样本的概率高，哪些样本的概率低，概率低的样本很大程度上就是异常值。
'''

#选择阈值 Threshold
#yval为样本概率 pval为计算出的概率
def selectThreshold(yval,pval):
    def computeF1(yval,pval):
        m=len(yval)
        tp=float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp=float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn=float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec=tp/(tp+fp) if (tp+fp) else 0
        rec=tp/(tp+fn)  if (tp+fn) else 0
        F1=2*prec*rec/(prec+rec) if (prec+rec) else 0
        return F1
    
    epsilons=np.linspace(min(pval),max(pval),1000)
    bestF1,bestEpsilon=0,0
    for e in epsilons:
        pval_=pval < e
        thisF1=computeF1(yval,pval_)
        if thisF1>bestF1:
            bestF1=thisF1
            bestEpsilon=e
    return bestF1,bestEpsilon

mu,sigma=getGaussianParams(x,useMultivariate=False)

y_pred=gaussian(x,mu,sigma)
yval_pred=gaussian(xval,mu,sigma)

bestF1,bestEpsilon=selectThreshold(yval,yval_pred)
print(bestF1,bestEpsilon)

xx=np.array([x[i] for i in range(len(y_pred)) if y_pred[i]<bestEpsilon])#异常值
#xx=[x[i] for i in range(len(y_pred)) if y_pred[i]<bestEpsilon]
print(len(xx))

plotdata()
plotContours(mu,sigma)
plt.scatter(xx[:,0],xx[:,1],facecolors='none',edgecolors='r')


# You should see a value epsilon of about 1.38e-18, and 117 anomalies found.