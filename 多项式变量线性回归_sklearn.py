# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:45:58 2019

@author: Q
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

path='E:\Coursera-ML-using-matlab-python-master\ex5data1'
data=loadmat(path)

xtrain,ytrain=data['X'],data['y']
xval,yval=data['Xval'],data['yval']
xtest,ytest=data['Xtest'],data['ytest']

plt.scatter(xtrain,ytrain)
plt.show()
#print(xtrain.shape,ytrain.shape,xval.shape,yval.shape,xtest.shape,ytest.shape)
#(12, 1) (12, 1) (21, 1) (21, 1) (21, 1) (21, 1)

#拟合线性回归
linear_regressor=LinearRegression()
linear_regressor.fit(xtrain,ytrain)
x2=np.linspace(-60,60,100).reshape(-1,1)
y2=linear_regressor.predict(x2)
plt.plot(x2,y2,label='y = ax + c')
plt.legend()


#多项式回归
polynomial=PolynomialFeatures(degree=3)#最高到三次项
x_transformed=polynomial.fit_transform(xtrain)#x转变成多项式
#x_transformed.shape is (12,4)

#多项式回归拟合
poly_linear_model=LinearRegression()
poly_linear_model.fit(x_transformed,ytrain)

xx=np.linspace(-60,60,100).reshape(-1,1)
xx_transformed=polynomial.fit_transform(xx)
yy=poly_linear_model.predict(xx_transformed)
plt.plot(xx,yy,label='y = ax^3+ bx^2 + cx + d')
plt.legend()

xval=polynomial.transform(xval)
print(linear_regressor.score(xtrain,ytrain))
print(poly_linear_model.score(x_transformed,ytrain))
print(poly_linear_model.score(xval,yval))

#0.7132909220428962
#0.9908201779936682
#0.9286268506855062