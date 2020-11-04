#-*-coding:utf-8-*-
# ZOU Jingcheng mean-shift2.0 2020/7

"""Code Encapsulation
   Wang Can 2020 / 08 / 22 Due to my limited level, but also considering the universality of the model, I made some sentence deletion.
   Hope you to criticize and correct.
"""
# preparation
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
# Ignore useless warnings (Scipy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning, module='sklearn',  lineno=196)
#After each run, you get the same result as the notebook
np.random.seed(42)
# To make the plot of matplotlib better
#get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Set the way to save the pictures
PROJECT_ROOT_DIR = "."


def save_fig(fig_id, tight_layout=True):
    '''
 Just need to create a folder of images in the directory where clustering_ test_ 202007121.ipynb file is located, then run to save automatic pictures

    
    :param fig_id: picture name
    '''
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

# read data
df = pd.read_excel('主量与Fe比值-1.xlsx')
# interpolation
df.fillna(0, inplace=True)

#Separate the real classification label from the feature
data_df = df.drop(['TRUE VALUE'], axis = 1)
labels = df['TRUE VALUE']

#Preprocessing (- 1 no preprocessing)

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
# Get the number of data and the number of features
n_samples, n_features = data.shape
# Get the number of category labels
n_labels = len(np.unique(labels))


# Core function，mean-shift clustering
bandwidth=estimate_bandwidth(data, quantile=0.1, n_samples=len(data))
# Print average bandwidth
print('The bandwidth is:',bandwidth)
# Model training
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(data)
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(ms.labels_)
n_clusters_ = len(labels_unique)
print('number of estimated clusters: %d'%n_clusters_)
print('mean_shift_labels: ',labels_unique)
print('mean_shift_centers: ', cluster_centers )

# Write the results of clustering into the original table
data['ms_clustering_label'] = ms.labels_
# Export the original table as CSV
data.to_csv('result.csv')

#For the universality of the model, the loop comparison grid search is selected by default, and the gridsearchcv is annotated out.
# # Using GridSearchCV to automatically find the optimal parameters
# bw_list = np.arange(0.1,10,0.1)
# bw_list = [round(x,10) for x in bw_list]
# params = {'bandwidth':bw_list}
# cluster = MeanShift()
# ms_best_model = GridSearchCV(cluster, params, cv=5, scoring='adjusted_rand_score', verbose=1,n_jobs=-1)
# ms_best_model.fit(data, labels)
# t2 = time.time()
# print('The time is：%3.f'%(t2-t1))
# # Parameters of the best model
# print("Parameters of the best model",ms_best_model.best_params_)
# #The score of the best model was evaluated using the adjusted_ rand_ Score)
# print("Score of the best model",ms_best_model.best_score_)
# Use loop comparison grid to search
t3 = time.time()
ars = []
bw_list = []
num_clusters = []
max_ars = -1
best_bw = None
best_n_clusters = None
for bw in np.arange(0.2,3,0.1):
    bw = round(bw,3) # Keep 3 decimal places
    # The same adjusted Rand coefficient as GridSearchCV was used as the evaluation criteria
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
print('*'*20+'The optimal solution'+'*'*20)
print('The optimal bandwidth：',best_bw)
print('Maximum adjusted Rand coefficient：',max_ars)
print('The optimal number of depolarization centers：',best_n_clusters)
t4 = time.time()
print('The time is：%3.f'%(t4-t3))
# Preservation of Solutions
result = pd.DataFrame({'bandwidth':bw_list,'n_clusters':num_clusters,'adjusted_rand_score':ars})
print(result)


# Results evaluation and output
def get_marks(estimator, data, name=None):
    """
      To get the score, there are five kinds of actual classification information that are required to know the data set, and there are three kinds that are not required,
       refer to the readme.txt
         
    :param estimator: model
    :param name: original method
    :param data: Feature dataset
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


# mean-shift score
get_marks(ms, data, name='Mean Shift')


# Picture output
# It is expressed in the form of a bar chart. The abscissa is the content and the ordinate is the quantity. 
data.hist(bins=50, figsize=(20, 15))
save_fig('data_describe')
plt.show()


def plot_scores(min_bw, max_bw, step, data, labels):
    '''
    Draw three kinds of rating charts of different initialization methods
    
    :param min_bw: lower limit of the bandwidth
    :param max_bw: upper limit of the bandwidth
    :param step:   sampling step
    :param labels: real label dataset
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

# The feature is reduced from 11 dimensions to 2 dimensions by using general PCA
pca3 = PCA(n_components = 2)
reduced_data = pca3.fit_transform(data)
get_marks(MeanShift(bandwidth = 2.551 , bin_seeding=True), reduced_data, name="PCA-based ms")
plt.plot(reduced_data[:, 0], reduced_data[:, 1], '*',color = 'orange', markersize=5)
plt.show()
# bandwidth = 2.551, bin_seeding = True
reduced_ms = MeanShift(bandwidth = 2.551 , bin_seeding = True).fit(reduced_data)
core_samples, cluster_ids = np.unique(reduced_ms.labels_), reduced_ms.labels_
# In cluster_ids,-1 indicates that the corresponding point is a noise point
reduced_df = pd.DataFrame(np.c_[reduced_data, cluster_ids],columns = ['feature1','feature2','cluster_id'])
reduced_df['cluster_id'] = reduced_df['cluster_id'].astype('i2')
reduced_df.plot.scatter('feature1','feature2', s = 100,
    c = list(reduced_df['cluster_id']),cmap = 'rainbow',colorbar = True,
    alpha = 0.6,title = 'mean shift cluster result')
plt.show()
print(reduced_df)
# Add the dimension reduced label to the original data and output it
df['dbscan_label'] = reduced_df['cluster_id']
df.to_csv('reduced_result.csv')






