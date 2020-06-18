# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:47:27 2019

PCA来实现降维

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

path='E:\Coursera-ML-using-matlab-python-master\ex7data1.mat'
data=loadmat(path)
x=data['X']#x.shape(50,2)

#数据标准化
def featureNormalize(x):
    means=x.mean(axis=0)#计算每一列的平均值
    stds=x.std(axis=0,ddof=1)#总体标准差ddof=0，样本标准差ddof=1
    x_norm=(x-means)/stds
    return x_norm,means,stds

def pca(x):
    sigma=((x.T).dot(x))/len(x)
    U,S,V=np.linalg.svd(sigma)
    return U,S,V

x_norm,means,stds=featureNormalize(x)
U,S,V=pca(x_norm)

plt.figure(figsize=(8,5))
plt.scatter(x[:,0],x[:,1],facecolors='none',edgecolors='b')#空心圆
#画出U1，数据在U1轴的投影代表了原始数据的巨大部分信息,是第一主成分
plt.plot([means[0],means[0]+1.5*S[0]*U[0,0]],
         [means[1],means[1]+1.5*S[0]*U[0,1]],
         c='r',label='fist principal component')
plt.plot([means[0],means[0]+1.5*S[1]*U[1,0]],
         [means[1],means[1]+1.5*S[1]*U[1,1]],
         c='g',label='second principal component')
plt.grid()
plt.axis('equal')#让x y轴长度相等
plt.legend()

#用PCA进行降维
#对数据进行投影
def projectdata(x,U,K):
    z=x.dot(U[:,:K])
    return z

z=projectdata(x_norm,U,1)
#print(z.shape)     (50, 1)

#降维后的数据
def recoverdata(z,U,K):
    x_rec=z.dot(U[:,:K].T)
    return x_rec

x_rec=recoverdata(z,U,1)
#print(x_rec.shape)  (50,2)

#可视化投影
plt.figure(figsize=(7,5))
plt.scatter(x_norm[:,0],x_norm[:,1],facecolors='none',
                 edgecolors='b',label='original data points')
plt.scatter(x_rec[:,0],x_rec[:,1],facecolors='none',
                 edgecolors='r',label='pca reduced data points')
plt.title('example dataset: reduced dimension points shown')
plt.xlabel('x1 fearture normalized')
plt.ylabel('x2 fearture normalized')
plt.axis('equal')
plt.grid()

for i in range(x_norm.shape[0]):
    plt.plot([x_norm[i,0],x_rec[i,1]],[x_norm[i,1],x_rec[i,1]],'k--')
plt.legend()

data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex7faces.mat')
x=data['X']
#print(x.shape) (5000, 1024)

#数据可视化
def displaydata(x,row,col):
    fig,ax=plt.subplots(row,col,figsize=(8,8))
    for i in range(row):
        for each in range(col):
            ax[i][each].matshow(x[i*col+each].reshape(32,32).T,cmap='gray_r')
            ax[i][each].set_xticks([])
            ax[i][each].set_yticks([])
displaydata(x,10,10)

x_norm,means,stds=featureNormalize(x)
U,S,V=pca(x_norm)
#print(U.shape,S.shape,V.shape)   (1024, 1024) (1024,) (1024, 1024)
displaydata(U[:,:36].T,4,4)

z=projectdata(x_norm,U,K=36)
x_rec=recoverdata(z,U,K=36)
displaydata(x_rec,10,10)