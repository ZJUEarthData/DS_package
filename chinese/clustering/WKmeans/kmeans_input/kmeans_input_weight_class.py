# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:59:32 2020

@author: 王少泽
"""
import random
import numpy as np
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

class kmeans_input_weight:
   
    def __check_params(self, data, k, weights, max_iter, tol):
        if k <= 0 or k > data.shape[0]:
            raise ValueError("k must be > 0 and <= {}, got {}".format(data.shape[0], k))

        if weights.size != data.shape[1]:
            raise ValueError("weights length expected {}, got {}".format(data.shape[0], len(weights)))

        if max_iter <= 0:
            raise ValueError("max_iter must be > 0, got {}".format(max_iter))

        if tol < 0.0:
            raise ValueError("tol must be >= 0.0, got {}".format(tol))
    
    def sqrsum(self, x):
        return np.sum(x * x)

    # 评价标准，测试用
    def get_marks(self, data, true_labels, predicted_labels):
        """获取评分，有五种需要知道数据集的实际分类信息，参考readme.txt
        :data: 待分析数据
        :true_labels: 真正分类标签
        :predicted_labels: 模型预测分类标签
        """
        print(30 * '*', "model performance", 30 * '*')
        print("Homogeneity Score         (均一性): ", metrics.homogeneity_score(true_labels, predicted_labels))
        print("Completeness Score        (完整性): ", metrics.completeness_score(true_labels, predicted_labels))
        print("V-Measure Score           (V量): ", metrics.v_measure_score(true_labels, predicted_labels))
        print("Adjusted Rand Score       (调整后兰德指数): ", metrics.adjusted_rand_score(true_labels, predicted_labels))
        print("Adjusted Mutual Info Score(调整后的共同信息): ", metrics.adjusted_mutual_info_score(true_labels, predicted_labels))
        print("Calinski Harabasz Score:  (方差比指数) ", metrics.calinski_harabasz_score(data, predicted_labels))
        print("Silhouette Score          (轮廓分数): ", metrics.silhouette_score(data, predicted_labels))
        
    def plus_plus(self, ds, k, random_state=42):
        """
        Create cluster centroids using the k-means++ algorithm.
        Parameters
        ----------
        ds : numpy array
            The dataset to be used for centroid initialization.
        k : int
            The desired number of clusters for which centroids are required.
        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        codes taken from: https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html
        """

        np.random.seed(random_state)
        randidx=random.randint(0,ds.shape[0])
        centroids = [ds[randidx]]

        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
            probs = dist_sq/dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
        
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
        
            centroids.append(ds[i])

        return np.array(centroids)