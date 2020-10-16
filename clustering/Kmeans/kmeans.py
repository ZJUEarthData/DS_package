#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('4.xlsx')
data = df.drop('O7', axis = 1)
labels = df['O7']

# 核心功能，非监督学习的网格搜索，请在这个基础上添加需要的功能，最后修改函数名以防冲突
def KmeansGridsearch(dmodel, data, param_dict):
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


kmeans = KMeans()
# 选择要测试的参数
kmeans_dict = {'n_clusters':[3,4,5],
               'init':['k-means++','random']}
output = KmeansGridsearch(kmeans,data,kmeans_dict)

# 评价标准，测试用
def get_marks(estimator, data, name=None):
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

# 测试结果
for i in range(len(output)):
    get_marks(output[i], data=data, name="output"+ str(i))

# 将测试结果绘制成图像，便于比较
def plotit(estimator, data):
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

plotit(output,data)





