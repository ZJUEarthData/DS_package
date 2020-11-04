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

#The core function, the grid search of unsupervised learning. Please add the required functions on this basis, and finally modify the function name to prevent conflicts
def KmeansGridsearch(dmodel, data, param_dict):
    """
    dmodel: default model
    data：training data
    labels: real classification
    param_dict: hyperparameter combination dictionary
    """
    output_models = []
    # create parameter grid
    # create hyperparametric grid
    param_grid = ParameterGrid(param_dict)
    
    # change the parameter attributes in dbscan according to the param_grid
       # modify the corresponding parameters of DBSCAN object according to the grid hyperparameters, train the model, and get the output data 
    for param in param_grid:
        for key, value in param.items():
            setattr(dmodel,key,value)
        dmodel.fit(data)
        model = clone(dmodel)
        output_models.append(model)
    # If you have other data to output, just add it  
    return (output_models)


kmeans = KMeans()
# select the parameters to be tested
kmeans_dict = {'n_clusters':[3,4,5],
               'init':['k-means++','random']}
output = KmeansGridsearch(kmeans,data,kmeans_dict)

# Evaluation criteria for testing
def get_marks(estimator, data, name=None):
    """     To get the score, there are five kinds of actual classification information that are required to know the data set, and there are three kinds that are not required,
       refer to the readme.txt
       
    :param estimator: model
    :param name: initial method
    :param data: feature data set
    """
    estimator.fit(data.astype(np.float64))
    print(30 * '*', name, 30 * '*')
    print(" Model and parameters     : ", estimator )
    print("Homogeneity Score         : ", metrics.homogeneity_score(labels, estimator.labels_))
    print("Completeness Score        : ", metrics.completeness_score(labels, estimator.labels_))
    print("V-Measure Score           : ", metrics.v_measure_score(labels, estimator.labels_))
    print("Adjusted Rand Score       : ", metrics.adjusted_rand_score(labels, estimator.labels_))
    print("Adjusted Mutual Info Score: ", metrics.adjusted_mutual_info_score(labels, estimator.labels_))
    print("Calinski Harabasz Score:   ", metrics.calinski_harabasz_score(data, estimator.labels_))
    print("Silhouette Score          : ", metrics.silhouette_score(data, estimator.labels_))

# test results
for i in range(len(output)):
    get_marks(output[i], data=data, name="output"+ str(i))

#  The test results are drawn into images for easy comparison
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





