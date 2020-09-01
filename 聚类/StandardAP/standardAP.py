#-*-coding:utf-8-*-
# 何灿(Sany) AP3.0 2020/07/17
# 代码封装 王璨 2020/08/22 由于本人水平有限，同时也考虑到模型的普适性，做出了一些语句的删减
# 。
# 恳请各位批评指正。
from __future__ import division, print_function, unicode_literals
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns

# 准备工作
# 读取数据
df = pd.read_excel('Test_2.xlsx')
# 插值
df.fillna(0, inplace=True)
data_df = df.drop('TRUE VALUE', axis=1)
labels = df['TRUE VALUE'].copy()
np.unique(labels)
data = df.drop('TRUE VALUE', axis=1)
labels = df['TRUE VALUE'].copy()
np.unique(labels)
# 在每一次的运行后获得的结果与这个notebook的结果相同
np.random.seed(42)
# 设置图片及输出

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
# 设置保存图片的途径
PROJECT_ROOT_DIR = "."


def save_fig(fig_id, tight_layout=True):
    '''
    只需在clustering_test_202007121.ipynb文件所在目录处，建立一个images的文件夹，运行即可保存自动图片
    
    :param tight_layout:
    :param fig_id: 图片名称
    '''
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


warnings.filterwarnings(action="ignore", category=FutureWarning, module='sklearn', lineno=196)
data.hist(bins=50, figsize=(20, 15))
save_fig('data_describe')
plt.show()


# 核心功能，AP聚类
def data_process(X, choice):
    if choice == 0:
        X = preprocessing.RobustScaler().fit_transform(X)
    elif choice == 1:
        X = preprocessing.MinMaxScaler().fit_transform(X)
    elif choice == 2:
        X = preprocessing.StandardScaler().fit_transform(X)
    elif choice == -1:
        X = X
    return X


data = pd.DataFrame(data_process(np.array(data_df), 2))
# 获取数据的数量和特征的数量
n_samples, n_features = data.shape
# 获取分类标签的数量
n_labels = len(np.unique(labels))
# 使用AP聚类算法
af = AffinityPropagation(preference=-500, damping=0.8, random_state=0)
print("聚类结果为：", af.fit(data))
# 获取簇的坐标
cluster_centers_indices = af.cluster_centers_indices_
print("簇坐标为：")
print(cluster_centers_indices)
# 获取分类的类别数量
af_labels = af.labels_
np.unique(af_labels)
params = {'preference': [-50, -100, -150, -200], 'damping': [0.5, 0.6, 0.7, 0.8, 0.9]}
cluster = AffinityPropagation(random_state=0)
af_best_model = GridSearchCV(cluster, params, cv=5, scoring='adjusted_rand_score', verbose=1, n_jobs=-1)
print(af_best_model.fit(data, labels))
# 获取最优模型
af1 = af_best_model.best_estimator_


# 定义评价函数
def get_marks(estimator, data, name=None, kmeans=None, af=None):
    """获取评分，有五种需要知道数据集的实际分类信息，有三种不需要，参考readme.txt
    
    :param estimator: 模型
    :param name: 初始方法
    :param data: 特征数据集
    """
    estimator.fit(data)
    print(20 * '*', name, 20 * '*')
    if kmeans:
        print("Mean Inertia Score: ", estimator.inertia_)
    elif af:
        cluster_centers_indices = estimator.cluster_centers_indices_
        print("The estimated number of clusters: ", len(cluster_centers_indices))
    print("Homogeneity Score: ", metrics.homogeneity_score(labels, estimator.labels_))
    print("Completeness Score: ", metrics.completeness_score(labels, estimator.labels_))
    print("V Measure Score: ", metrics.v_measure_score(labels, estimator.labels_))
    print("Adjusted Rand Score: ", metrics.adjusted_rand_score(labels, estimator.labels_))
    print("Adjusted Mutual Info Score: ", metrics.adjusted_mutual_info_score(labels, estimator.labels_))
    print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(data, estimator.labels_))
    print("Silhouette Score: ", metrics.silhouette_score(data, estimator.labels_))


# 结果评价与输出
get_marks(af, data=data, af=True)
# 将AP聚类聚类的结果写入原始表格中
df['ap_clustering_label'] = af.labels_
# 以csv形式导出原始表格
df.to_csv('test2_result.csv')
# 最后两列为两种聚类算法的分类信息
print("最优模型的参数：", af_best_model.best_params_)
# 最优模型的评分
get_marks(af1, data=data, af=True)


# 参数-preference,damping对结果得分可视化
def plot_scores(preference, damping, data, labels):
    i = []
    y_silhouette_scores = []
    y_calinski_harabaz_scores = []
    preference = [round(x, 3) for x in preference]
    damping = [round(x, 3) for x in damping]
    for m in preference:
        for k in damping:
            ap_model = AffinityPropagation(preference=m, damping=k,random_state=None)
            pred = ap_model.fit(data)
            i.append(k)
            if len(np.unique(pred.predict(data))) == 1:
                y_silhouette_scores.append(-1)
                y_calinski_harabaz_scores.append(-1)
            else:
                y_silhouette_scores.append(silhouette_score(data, pred.predict(data)))
                y_calinski_harabaz_scores.append(calinski_harabasz_score(data, pred.predict(data)))
    y_silhouette_scores = np.array(y_silhouette_scores).reshape(len(preference), len(damping))
    y_calinski_harabaz_scores = np.array(y_calinski_harabaz_scores).reshape(len(preference), len(damping))
    new = [y_silhouette_scores, y_calinski_harabaz_scores]
    for j in range(len(new)):
        plt.figure(j + 1)
        result = pd.DataFrame(new[j])
        result = result.rename(columns=pd.Series(damping), index=pd.Series(preference))

        fig, ax = plt.subplots(figsize=(15, 15))

        if j == 0:
            name = 'silhouette scores'
            sns.heatmap(result, annot=True, vmax=1, vmin=-1, xticklabels=True, yticklabels=True, square=True,
                        cmap="rainbow")
        elif j == 1:
            name = 'calinski harabaz scores'
            sns.heatmap(result, annot=True, xticklabels=True, yticklabels=True, square=True, cmap="rainbow")
        plt.ylabel('preference')
        plt.xlabel('damping')
        plt.title('{}_scores'.format(name))
        save_fig('{}'.format(name))


plot_scores(np.arange(-200, -50, 50), np.arange(0.5, 1, 0.1), data, labels)
plt.show()