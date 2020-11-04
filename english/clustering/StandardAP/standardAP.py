#-*-coding:utf-8-*-
# HE Can(Sany) AP3.0 2020/07/17

"""Code Encapsulation
   Wang Can 2020 / 08 / 22 Due to my limited level, but also considering the universality of the model, I made some sentence deletion.
   Hope you to criticize and correct.
"""


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

# preparation
# read data
df = pd.read_excel('Test_2.xlsx')
# interpolation
df.fillna(0, inplace=True)
data_df = df.drop('TRUE VALUE', axis=1)
labels = df['TRUE VALUE'].copy()
np.unique(labels)
data = df.drop('TRUE VALUE', axis=1)
labels = df['TRUE VALUE'].copy()
np.unique(labels)
# After each run, you get the same result as the notebook.
np.random.seed(42)
# Set up pictures and output

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
# Set the way to save the pictures
PROJECT_ROOT_DIR = "."


def save_fig(fig_id, tight_layout=True):
    '''
    Just need to create a folder of images in the directory where clustering_ test_ 202007121.ipynb file is located, then run to save automatic pictures

    
    :param tight_layout:
    :param fig_id: picture name
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


# Core function, AP clustering
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
# Get the number of data and the number of features
n_samples, n_features = data.shape
# Get the number of category labels 
n_labels = len(np.unique(labels))
# Use AP clustering algorithm
af = AffinityPropagation(preference=-500, damping=0.8, random_state=0)
print("聚类结果为：", af.fit(data))
# Get the coordinates of the cluster
cluster_centers_indices = af.cluster_centers_indices_
print("簇坐标为：")
print(cluster_centers_indices)
# Gets the number of categories classified
af_labels = af.labels_
np.unique(af_labels)
params = {'preference': [-50, -100, -150, -200], 'damping': [0.5, 0.6, 0.7, 0.8, 0.9]}
cluster = AffinityPropagation(random_state=0)
af_best_model = GridSearchCV(cluster, params, cv=5, scoring='adjusted_rand_score', verbose=1, n_jobs=-1)
print(af_best_model.fit(data, labels))
# Get the best model
af1 = af_best_model.best_estimator_


# Define evaluation function
def get_marks(estimator, data, name=None, kmeans=None, af=None):
    """
    To get the score, there are five kinds of actual classification information that are required to know the data set, and there are three kinds that are not required,
       refer to the readme.txt
       
    :param estimator: model
    :param name: initial method
    :param data: feature data set
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


# Results evaluation and output
get_marks(af, data=data, af=True)
# Write the results of AP clustering into the original table
df['ap_clustering_label'] = af.labels_
# Export the original table as csv
df.to_csv('test2_result.csv')
# The last two columns are the classification information of the two clustering algorithms
print("最优模型的参数：", af_best_model.best_params_)
# The score of the best model
get_marks(af1, data=data, af=True)


#Parameters - preference and damping are used to visualize the result score
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
