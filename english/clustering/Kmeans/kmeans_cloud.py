# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:53:34 2020

@author: Administrator
"""
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
import sys

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



   # Evaluation criteria for testing
def get_marks(estimator, data, labels, name=None):
    """To get the score, there are five kinds of actual classification information that are required to know the data set, and there are three kinds that are not required,
       refer to the readme.txt
       
    :param estimator: model
    :param name: initial method
    :param data: feature data set
    """
    estimator.fit(data.astype(np.float64))
    print(30 * '*', name, 30 * '*')
    print("Model and parameters      : ", estimator )
    print("Homogeneity Score         : ", metrics.homogeneity_score(labels, estimator.labels_))
    print("Completeness Score        : ", metrics.completeness_score(labels, estimator.labels_))
    print("V-Measure Score           : ", metrics.v_measure_score(labels, estimator.labels_))
    print("Adjusted Rand Score       : ", metrics.adjusted_rand_score(labels, estimator.labels_))
    print("Adjusted Mutual Info Score: ", metrics.adjusted_mutual_info_score(labels, estimator.labels_))
    print("Calinski Harabasz Score   :   ", metrics.calinski_harabasz_score(data, estimator.labels_))
    print("Silhouette Score          : ", metrics.silhouette_score(data, estimator.labels_))


#  The test results are drawn into images for easy comparison
def plotit(estimator, data, labels):
    
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


def read_para(FEATURE_FILE_PATH):
    para = pd.read_excel('para.xlsx', header=None, dtype='object')
    dic = para.set_index(0).T.to_dict('list')
    for i in dic:
        dic[i] = [x for x in dic[i] if x == x]
    return dic

def plot_learning_curve(model,data,labels,OUTPUT_RESULTS):
    train_sizes, train_scores, test_scores = learning_curve(model, data, labels,
                                                            scoring='adjusted_rand_score', cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)  # To average the training score set by row 
    train_scores_std = np.std(train_scores, axis=1)   #  Calculate the standard deviation of training matrix
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()  # Set the background to gridlines

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    # plt.fill_between(function fills the space of the upper and lower variances of the average model accuracy with colors.
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        # Then use plt.plot The () function draws the average of the model accuracy
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross_validation score')
    plt.legend(loc='best')  # Show legend
    #plt.show()
    plt.savefig(OUTPUT_RESULTS + 'training_results.png')
    
def main():
    
    FEATURE_FILE_PATH = sys.argv[1]
    DATA_FILE_PATH = sys.argv[2]
    OUTPUT_MODEL = sys.argv[3]
    OUTPUT_RESULTS = sys.argv[4]
    
    df = pd.read_excel('test4.xlsx')
    data = df.drop('TRUE VALUE', axis=1)
    labels = df['TRUE VALUE']
    kmeans = KMeans()
    
  # select the parameters to be tested
    ap_dict = read_para(FEATURE_FILE_PATH)
    output = KmeansGridsearch(kmeans,data,ap_dict)

    # test results
    #for i in range(len(output)):
    #    get_marks(output[i], data=data,labels=labels, name="output" + str(i))
        
    #plotit(output, data, labels)
    
    # test results
    for i in range(len(output)):
        get_marks(output[i], data=data, labels=labels, name="output" + str(i))
    af_best_model = GridSearchCV(kmeans, ap_dict, cv=5, scoring='adjusted_rand_score', verbose=1, n_jobs=-1)
    af_result = af_best_model.fit(data, labels)
    print(af_result.best_params_)
    
        # save model
    joblib.dump(af_best_model.best_estimator_, OUTPUT_MODEL+ "./test.pkl")

    # save parameters
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")
    result = pd.DataFrame(af_result.best_params_, index=['value'])
    result.to_csv(OUTPUT_RESULTS+"{}.csv".format(TIMESTAMP), index=None)

    # draw learning curves
    #plt.subplot(3,3,8)
    plot_learning_curve(af_best_model.best_estimator_,data,labels,OUTPUT_RESULTS)

if __name__ == '__main__':
    main()



