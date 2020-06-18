# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:29:36 2019

选择C和sigma参数
@author: Q
"""

import numpy as np
from scipy.io import loadmat
from sklearn import svm
from functions import plotdata
from functions import plotboundary

data=loadmat('E:\Coursera-ML-using-matlab-python-master\ex6data3.mat')
x=data['X']
y=data['y']
xval,yval=data['Xval'],data['yval']
#x.shape,y.shape,xval.shape,yval.shape
#(211, 2) (211, 1) (200, 2) (200, 1)

Cvalues=(0.01,0.03,0.1,0.3,1.,3.,10.,30.)
sigmavalues=Cvalues
best_pair,best_score=(0,0),0

#选择SVM的C参数和sigma参数
for C in Cvalues:
    for sigma in sigmavalues:
        gamma=np.power(sigma,-2)/2
        model=svm.SVC(C=C,kernel='rbf',gamma=gamma)
        model.fit(x,y.flatten())
        score=model.score(xval,yval)
        if score>best_score:
            best_score=score
            best_pair=(C,sigma)
print('best_score={},best_pair={}'.format(best_score,best_pair))


model=svm.SVC(C=1.0,kernel='rbf',gamma=np.power(.1,-2)/2)
model.fit(x,y.flatten())#y都应该加上flatten()
plotdata(x,y)
plotboundary(model,x)