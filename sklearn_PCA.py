 # -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:13:15 2019

@author: qo
"""


from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from timeit import timeit

x,y=load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y)

def exec_without_pca():
    knn_clf=KNeighborsClassifier()
    knn_clf.fit(x_train,y_train)
    print(knn_clf.score(x_test,y_test))

def exec_with_pca():
    knn_clf=KNeighborsClassifier()
    #n_components: int, float, None 或 string，PCA算法中所要保留的主成分个数，也即保留下来的特征个数
    # 如果 n_components = 1，将把原始数据降到一维；
    #我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个(0,1]之间的浮点数。
    # 如果赋值为string，如n_components='mle'，将自动选取特征个数，使得满足所要求的方差百分比；
    # 如果没有赋值，默认为None，特征个数不会改变（特征数据本身会改变）。
    pca=PCA(n_components=0.95)
    pca.fit(x_train,y_train)
    x_train_dunction=pca.transform(x_train)
    x_test_dunction=pca.transform(x_test)
    knn_clf.fit(x_train_dunction,y_train)
    print (knn_clf.score(x_test_dunction,y_test))

if __name__=='__main__':
    print('Time of method[exec_without_pca] costs:',timeit('exec_without_pca()',setup='from __main__ import exec_without_pca',number=3))
    print('Time of method[exec_with_pac] costs:',timeit('exec_with_pca()',setup='from __main__ import exec_with_pca',number=3))