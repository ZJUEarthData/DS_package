# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 11:01:26 2020

@author: Administrator
"""
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import GridSearchCV,ParameterGrid
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import joblib
import datetime

class GeoKmeans:

# 核心功能，非监督学习的网格搜索，请在这个基础上添加需要的功能，最后修改函数名以防冲突
    def KmeansGridsearch(self, dmodel, data, param_dict):
        """
        dmodel: 默认模型
        data：训练数据
        labels: 真实分类
        param_dict: 超参数组合字典
        """
        output_models = []
        # create parameter grid
        # 构建超参数网格
        param_grid = ParameterGrid(param_dict)
    
        # change the parameter attributes in dbscan according to the param_grid
        # 依据网格超参数修改dbscan object 的对应参数，训练模型，得出输出数据
        for param in param_grid:
            for key, value in param.items():
                setattr(dmodel,key,value)
            dmodel.fit(data)
            model = clone(dmodel)
            output_models.append(model)
        # 如果有其他需要输出的数据，继续往里面添加就可以   
        return (output_models)



    # 评价标准，测试用
    def get_marks(self, estimator, data, labels, name=None):
        """获取评分，有五种需要知道数据集的实际分类信息，有三种不需要，参考readme.txt
    
        :param estimator: 模型
        :param name: 初始方法
        :param data: 特征数据集
        """
        estimator.fit(data.astype(np.float64))
        print(30 * '*', name, 30 * '*')
        print("       模型及参数: ", estimator )
        print("Homogeneity Score         (均一性): ", metrics.homogeneity_score(labels, estimator.labels_))
        print("Completeness Score        (完整性): ", metrics.completeness_score(labels, estimator.labels_))
        print("V-Measure Score           (V量): ", metrics.v_measure_score(labels, estimator.labels_))
        print("Adjusted Rand Score       (调整后兰德指数): ", metrics.adjusted_rand_score(labels, estimator.labels_))
        print("Adjusted Mutual Info Score(调整后的共同信息): ", metrics.adjusted_mutual_info_score(labels, estimator.labels_))
        print("Calinski Harabasz Score:  (方差比指数) ", metrics.calinski_harabasz_score(data, estimator.labels_))
        print("Silhouette Score          (轮廓分数): ", metrics.silhouette_score(data, estimator.labels_))


    # 将测试结果绘制成图像，便于比较
    def plotit(self, estimator, data, labels):
    
        plt.subplot(3,3,1)
        plt.subplots_adjust(0,0,2,2)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.homogeneity_score(labels, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Homogeneity Score')
        plt.subplot(3,3,2)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.completeness_score(labels, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Completeness Score')
        plt.subplot(3,3,3)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.v_measure_score(labels, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('V-Measure Score')
        plt.subplot(3,3,4)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.adjusted_rand_score(labels, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Adjusted Rand Score')
        plt.subplot(3,3,5)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.adjusted_mutual_info_score(labels, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Adjusted Mutual Info Score')
        plt.subplot(3,3,6)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.calinski_harabasz_score(data, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Calinski Harabasz Score')
        plt.subplot(3,3,7)
        home = []
        for i in range(len(estimator)):
            home.append(metrics.silhouette_score(data, estimator[i].labels_))
            plt.axvline(x=i,linestyle='--',linewidth=1,color='red')
        plt.plot(home)
        plt.title('Silhouette Score')


    def read_para(self):
        para = pd.read_excel('para.xlsx', header=None, dtype='object')
        dic = para.set_index(0).T.to_dict('list')
        for i in dic:
            dic[i] = [x for x in dic[i] if x == x]
        return dic

    def plot_learning_curve(self, model,data,labels):
        train_sizes, train_scores, test_scores = learning_curve(model, data, labels,
                                                            scoring='adjusted_rand_score', cv=5)
        train_scores_mean = np.mean(train_scores, axis=1)  # 将训练得分集合按行的到平均值
        train_scores_std = np.std(train_scores, axis=1)  # 计算训练矩阵的标准方差
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()  # 背景设置为网格线

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                         color='r')
        # plt.fill_between()函数会把模型准确性的平均值的上下方差的空间里用颜色填充。
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        # 然后用plt.plot()函数画出模型准确性的平均值
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross_validation score')
        plt.legend(loc='best')  # 显示图例
        plt.show()
    




