#-*-coding:utf-8-*-
# 邹景成 mean-shift2.0 2020/7
# 代码封装 王璨 2020/08/22 由于本人水平有限，同时也考虑到模型的普适性，做出了一些语句的删减
# 没有任何冒犯的意思，恳请各位作者原谅，批评指正。
# 准备工作
from __future__ import division, print_function, unicode_literals
import numpy as np
from IPython import get_ipython
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing,metrics
import os 
import time
import matplotlib as mpl
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn import metrics
# 忽略掉没用的警告 (Scipy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning, module='sklearn',  lineno=196)
# 在每一次的运行后获得的结果与这个notebook的结果相同
np.random.seed(42)
# 让matplotlib的图效果更好
#get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 设置保存图片的途径
PROJECT_ROOT_DIR = "."


def save_fig(fig_id, tight_layout=True):
    '''
    只需在clustering_test_202007121.ipynb文件所在目录处，建立一个images的文件夹，运行即可保存自动图片
    
    :param fig_id: 图片名称
    '''
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

# 读取数据集
df = pd.read_excel('主量与Fe比值-1.xlsx')
# 插值
df.fillna(0, inplace=True)
# 将真实的分类标签与特征分开
data_df = df.drop(['TRUE VALUE'], axis = 1)
labels = df['TRUE VALUE']
# 预处理(-1不做预处理)


def data_process(X,choice):
    if choice==0:
        X=preprocessing.RobustScaler().fit_transform(X)
    elif choice==1:
        X=preprocessing.MinMaxScaler().fit_transform(X)
    elif choice==2:
        X=preprocessing.StandardScaler().fit_transform(X)
    elif choice==-1:
        X=X
    return X


data = pd.DataFrame(data_process(np.array(data_df), 2))
# 获取数据的数量和特征的数量
n_samples, n_features = data.shape
# 获取分类标签的数量
n_labels = len(np.unique(labels))


# 核心功能，mean-shift聚类
bandwidth=estimate_bandwidth(data, quantile=0.1, n_samples=len(data))
# 打印平均带宽
print('带宽为:',bandwidth)
# 模型训练
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(ms.labels_)
n_clusters_ = len(labels_unique)
print('number of estimated clusters: %d'%n_clusters_)
print('mean_shift_labels: ',labels_unique)
print('mean_shift_centers: ', cluster_centers )

# 将聚类的结果写入原始表格中
data['ms_clustering_label'] = ms.labels_
# 以csv形式导出原始表格
data.to_csv('result.csv')

# 为了模型的普适性，默认选择使用循环对比网格搜索，将GridSearchCV注释掉。
# t1 = time.time()
# # 使用GridSearchCV自动寻找最优参数
# bw_list = np.arange(0.1,10,0.1)
# bw_list = [round(x,10) for x in bw_list]
# params = {'bandwidth':bw_list}
# cluster = MeanShift()
# ms_best_model = GridSearchCV(cluster, params, cv=5, scoring='adjusted_rand_score', verbose=1,n_jobs=-1)
# ms_best_model.fit(data, labels)
# t2 = time.time()
# print('用时：%3.f'%(t2-t1))
# # 最优模型的参数
# print("最优模型的参数",ms_best_model.best_params_)
# # 最优模型的评分，使用调整的兰德系数(adjusted_rand_score)作为评分
# print("最优模型的评分",ms_best_model.best_score_)
# 使用循环对比网格搜索
t3 = time.time()
ars = []
bw_list = []
num_clusters = []
max_ars = -1
best_bw = None
best_n_clusters = None
for bw in np.arange(0.2,3,0.1):
    bw = round(bw,3) # 保留3位小数
    # 使用与GridSearchCV相同的调整兰德系数作位评价标准
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(data)
    tmp = metrics.adjusted_rand_score(labels,ms.labels_)
    ars.append(tmp)
    bw_list.append(bw)
    num_clusters.append(len(ms.cluster_centers_))
    if max_ars < tmp:
        max_ars = tmp
        best_bw = bw
        best_n_clusters = len(ms.cluster_centers_)
print('*'*20+'最优解'+'*'*20)
print('最佳带宽：',best_bw)
print('最高调整兰德系数：',max_ars)
print('最佳解簇中心数：',best_n_clusters)
t4 = time.time()
print('用时：%3.f'%(t4-t3))
# 保存解的情况
result = pd.DataFrame({'bandwidth':bw_list,'n_clusters':num_clusters,'adjusted_rand_score':ars})
print(result)


# 结果评价与输出
def get_marks(estimator, data, name=None):
    """获取评分，有五种需要知道数据集的实际分类信息，有三种不需要，参考readme.txt
    
    :param estimator: 模型
    :param name: 初始方法
    :param data: 特征数据集
    """
    estimator.fit(data)
    print(20 * '*', name, 20 * '*')
    
    print("Homogeneity Score: ", metrics.homogeneity_score(labels, estimator.labels_))
    print("Completeness Score: ", metrics.completeness_score(labels, estimator.labels_))
    print("V Measure Score: ", metrics.v_measure_score(labels, estimator.labels_))
    print("Adjusted Rand Score: ", metrics.adjusted_rand_score(labels, estimator.labels_))
    print("Adjusted Mutual Info Score: ", metrics.adjusted_mutual_info_score(labels, estimator.labels_))
    print("Calinski Harabasz Score: ", metrics.calinski_harabasz_score(data, estimator.labels_))
    print("Silhouette Score: ", metrics.silhouette_score(data, estimator.labels_))


# mean-shift的得分情况
get_marks(ms, data, name='Mean Shift')


# 图片输出
# 用柱状图的方式表示出来，横坐标为含量，纵坐标为数量
data.hist(bins=50, figsize=(20, 15))
save_fig('data_describe')
plt.show()


def plot_scores(min_bw, max_bw, step, data, labels):
    '''画出不同初始化方法的三种评分图
    
    :param min_bw: 带宽下限
    :param max_bw: 带宽上限
    :param step:   取样步长
    :param labels: 真实标签的数据集
    '''
    
    i = []
    y_silhouette_scores = []
    y_calinski_harabaz_scores = []
    
    bw_list = np.arange(min_bw,max_bw,step)
    bw_list = [round(x,3) for x in bw_list]
    for k in bw_list:
        ms_model = MeanShift(bandwidth=k, bin_seeding=True)
        pred = ms_model.fit_predict(data)
        i.append(k)
        y_silhouette_scores.append(silhouette_score(data, pred))
        y_calinski_harabaz_scores.append(calinski_harabasz_score(data, pred))
    plt.figure(1)
    plt.plot(result.bandwidth, result.n_clusters,'b-')
    plt.xlabel('bandwidth')
    plt.ylabel('n_clusters')
    plt.title('n_clusters vs bandwidth')
    save_fig('n_clusters vs bandwidth')
    plt.figure(2)
    plt.plot(i,y_calinski_harabaz_scores,'b-')
    plt.xlabel('bandwidth')
    plt.ylabel('calinski_harabasz_scores with mean shift')
    plt.title('calinski_harabasz with mean shift')
    save_fig('calinski_harabasz with mean shift')
    plt.figure(3)
    plt.plot(i,y_silhouette_scores,'b-')
    plt.xlabel('bandwidth')
    plt.ylabel('silhouette_scores with mean shift')
    save_fig('silhouette with mean shift')
    plt.title('silhouette with mean shift')
    plt.figure(4)
    plt.plot(result.bandwidth, result.adjusted_rand_score, 'b-')
    plt.xlabel('bandwidth')
    plt.ylabel('adjusted_rand_score')
    save_fig('adjusted_rand_score with mean shift')
    plt.title('adjusted_rand_score with mean shift')


plot_scores(1, 5, 0.25, data, labels)
plt.show()

# 使用普通PCA进行降维，将特征从11维降至2维
pca3 = PCA(n_components = 2)
reduced_data = pca3.fit_transform(data)
get_marks(MeanShift(bandwidth = 2.551 , bin_seeding=True), reduced_data, name="PCA-based ms")
plt.plot(reduced_data[:, 0], reduced_data[:, 1], '*',color = 'orange', markersize=5)
plt.show()
# bandwidth = 2.551, bin_seeding = True
reduced_ms = MeanShift(bandwidth = 2.551 , bin_seeding = True).fit(reduced_data)
core_samples, cluster_ids = np.unique(reduced_ms.labels_), reduced_ms.labels_
# cluster_ids中-1表示对应的点为噪声点
reduced_df = pd.DataFrame(np.c_[reduced_data, cluster_ids],columns = ['feature1','feature2','cluster_id'])
reduced_df['cluster_id'] = reduced_df['cluster_id'].astype('i2')
reduced_df.plot.scatter('feature1','feature2', s = 100,
    c = list(reduced_df['cluster_id']),cmap = 'rainbow',colorbar = True,
    alpha = 0.6,title = 'mean shift cluster result')
plt.show()
print(reduced_df)
# 将降维后的label并到原数据并输出
df['dbscan_label'] = reduced_df['cluster_id']
df.to_csv('reduced_result.csv')






