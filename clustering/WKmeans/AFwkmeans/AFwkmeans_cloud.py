# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 20:44:36 2020

@author: 王少泽
"""
import random
import numpy as np
from sklearn import metrics
import pandas as pd
from numpy import mean, std
import matplotlib.pyplot as plt
import datetime
import sys


def CalcWeight(x):
    m, n = x.shape
    s = np.zeros((1,1), dtype=float)
    cv = np.zeros((1,n), dtype=float)
    weight = np.zeros((1,n), dtype=float)
    
    for i in range(n):
        cv[0][i] = std(x[:,i])/mean(x[:,i])
        
    s = np.sum(cv, axis=1)
    
    for j in range(n):
        weight[0][j] = cv[0][j] / s
    return weight

def initCentroid(x,k):
    n = np.size(x,0)
    idx_rand = np.array(random.sample(range(1,n), k))
    centroid = x[idx_rand, :]
    return centroid

def __check_params(data, k, weights, max_iter, tol):
    if k <= 0 or k > data.shape[0]:
        raise ValueError("k must be > 0 and <= {}, got {}".format(data.shape[0], k))

    if weights.size != data.shape[1]:
        raise ValueError("weights length expected {}, got {}".format(data.shape[0], len(weights)))

    if max_iter <= 0:
        raise ValueError("max_iter must be > 0, got {}".format(max_iter))

    if tol < 0.0:
        raise ValueError("tol must be >= 0.0, got {}".format(tol))
        
def sqrsum(x):
    return np.sum(x * x)

# 评价标准，测试用
def get_marks(data, true_labels, predicted_labels):
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

def plus_plus(ds, k, random_state=42):
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

def main():
    
    DATA_FILE_PATH = sys.argv[1]
    OUTPUT_RESULTS = sys.argv[2]
    OUTPUT_MODEL = sys.argv[3]
    
    orig_data = pd.read_excel(DATA_FILE_PATH)
    orig_data.dropna(inplace=True)

    x_orig_data = orig_data.drop('TRUE VALUE',axis=1)
    y_label = orig_data['TRUE VALUE']

    x_data=np.array(x_orig_data)

    n_iters = 10
    n_clusters = 4
    centroids = plus_plus(x_data, n_clusters)
    dist = np.zeros((x_data.shape[0],n_clusters), dtype=float)

    print("initial centroids are:",centroids)

    w = np.ones((1,x_data.shape[1]), dtype=float)/x_data.shape[1]  # weight initialization
    tol=1e-7
    count=0
    C = np.zeros(n_iters)
    __check_params(x_data, n_clusters, w, n_iters, tol)

    for i in range(n_iters):
        count+=1
        group= np.zeros(n_clusters)
        old_centroids=centroids.copy()
        for j in range(x_data.shape[0]):
            distance = np.power(x_data[j,:]-centroids, 2)  
            distance = np.sum(w*distance, axis=1)
            distance = np.sqrt(distance) 
            dist[j,:] = distance
        idx = np.argmin(dist, axis=1)   # predicted group index

        for k in range(n_clusters):
            d = x_data[idx==k, :]
            group[k]=d.shape[0]
            centroids[k,:] = np.mean(d, axis=0)

#     adaptive feature weighted 自适应权重重新分配       
        cc=np.zeros(d.shape[1])
        for l in range(d.shape[1]):
            ddn=0
            meanl=np.mean(centroids[:,l])
            ddm=np.power(centroids[:,l]-meanl, 2)
            ddm=np.sum(ddm)
            for k in range(n_clusters):
                dd=x_data[idx==k, :]
                dn=np.power(dd[:,l]-centroids[k,l], 2)
                ddn=ddn+np.sum(dn)
            cc[l]=ddm/ddn
        
        for n in range(d.shape[1]):
            w[0,n]=cc[n]/sum(cc)
            
        
        
        print('group result iteration', count, group)   # group number after each iterration
        centroid_change=sqrsum(centroids - old_centroids)
        C[i]=centroid_change
        if centroid_change <= tol:
                break
    print("AFwkmeans Finish!")

    plt.plot(np.arange(n_iters)+1, C, color='blue', marker='o', markersize=5, label='centroid distance change')
    plt.grid(True)
    plt.xlabel("Number of iteration")
    plt.ylabel("centroid distance change")
    plt.legend(loc='best')
    plt.savefig(OUTPUT_RESULTS+'training results.png')
    plt.show()

    print(get_marks(x_data, y_label, idx))


 # 保存参数
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")
    result = pd.DataFrame(w, index=['value'])
    result.to_csv(OUTPUT_MODEL+"AFwkmeans_weight_{}.csv".format(TIMESTAMP), index=None)

if __name__ == '__main__':
    main()
