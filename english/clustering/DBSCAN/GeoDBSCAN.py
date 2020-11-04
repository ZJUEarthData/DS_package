# -*- coding: utf-8 -*-
"""
Created on 17/08/2020

@author: Jingwei Liu

Version 1.1
"""

from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid 
from sklearn.base import clone
import pandas as pd
from MeasureScore import get_measure_scores
from MeasureScore import get_best_score_index


class GeoDBSCAN:
        
    # Initialization
    def __init__(self):
        """
    Initialization class

    Internal parameters
        --------
        model: the default model of DBSCAN
        clusters: used to store the number of clusters for training all models
        piece: used to store the number of samples in each cluster of all training models
        otherstates: other statistics, such as noise ratio
        outliers: used to store the number of noise points in all training models
        pred_labels : used to store prediction tags of all training models for training data
        score_measure : evaluation criteria used to select the best model (usually abbreviated as acronym) e.g. adjusted Rand index: ARC
        scores : used to store the evaluation scores of all models
        best_model_param : Hyperparameter of the best model of all models
        best_score :  the evaluation score of the best model
        best_index : index of the best model

        """
            
        self.model = DBSCAN()
        self.clusters = []
        self.piec = []  # points in each cluster (Number of samples per cluster)
        self.otherstats = {}
        self.outliers = []
        self.pred_labels = []
        self.score_measure = None
        self.scores = []
        self.best_model_param = {}
        self.best_score = None
        self.best_index = None


    #Grid search is constructed to obtain the effect of DBSCAN algorithm under different parameters
    #The data length of each key in the dictionary can be different
    def DBSCANGridsearch(self, data, param_dict, labels = None, measure = 'ARC', best = True):
        """
        Grid search of DBSCAN

        Input parameters
        ----------
        data : pandas dataframe
          Training data (without real tags)
        param_dict : dictionary
            A dictionary with hyperparameters which is used to build a search grid. The data length of each key in the dictionary can be different.
        labels : array-like of shape, optional
            Real data label, the default value is None.
        measure : string, optional
            Evaluation criteria, of which the default value is' ARC '-- adjust Rand index.
        best : boolean, optional
            Whether to find the best model, the default value is True.
            
        return value
        -------
        nothing

        """
        
        # Initialization (prevent the internal parameter from increasing infinitely)
        self.__init__()
        
        #  initial assignment
        dbscan = clone(self.model)
        self.score_measure = measure
        
        # Other class statistics initialization
        ndratio_dict = {'noise_data_ratio':[]}
        self.otherstats.update(ndratio_dict)  
           
        # Construction of hyperparametric grid
        param_grid = ParameterGrid(param_dict)
         
        # According to the grid hyperparameters, modify the corresponding parameters of dbscan object, train the model, and update the internal parameter.
        for param in param_grid:
            for key, value in param.items():
                setattr(dbscan,key,value)
       
            # train model
            model = clone(dbscan)
            model.fit(data)
            
            # Count the number of clusters (without noise) 
            clusters = set(model.labels_)
            clusters.remove(-1)           
            self.clusters.append(len(clusters))
             
            # Count the number of samples in each cluster (without noise), and cluster noise 
            label_sr = pd.Series(model.labels_)
            self.piec.append(label_sr[label_sr != -1].value_counts().values)
            self.outliers.append(label_sr[label_sr == -1].value_counts().values)
             
        
             # other statistical values
             # noise ratio
            ndratio = sum(model.labels_ == -1) / len(model.labels_)
            self.otherstats.get('noise_data_ratio').append(ndratio)
             
            # Storage model prediction label
            self.pred_labels.append(model.labels_)
             
            
        #If  best == True, the scores of all models are calculated according to the evaluation criteria.
        if best == True:
            self.score_measure = measure
            for pred_labels in self.pred_labels:
                score = get_measure_scores(measure, data, pred_labels, labels)  # Return the evaluation value of the model
                self.scores.append(score)
               
        # find the best model
        best_index = get_best_score_index(self.scores, measure)  # return the index of the best model
        self.best_model_param = pd.DataFrame(param_grid[best_index],index=[0])
        self.best_score = self.scores[best_index]
        self.best_index = best_index        
        
         # output information
        print("model training completed")
        print("The number of total training models{}".format(len(param_grid)))
  
    
    #Batch processing is used to obtain the effect of DBSCAN algorithm under different hyperparameters
    #The data length of each key in the dictionary must be the same
    def DBSCANBatchSearch(self, data, param_dict, labels = None, measure = 'ARC', best = True):
        """
        DBSCAN Batch Search

        Input parameters
        ----------
        data : pandas dataframe
            Training data (without real tags).
        param_dict : dictionary
            A dictionary with hyperparameters which is used to build a search grid. The data length of each key in the dictionary can be different.
        labels : array-like of shape, optional
              Real data label, the default value is None.
        measure : string, optional
              Evaluation criteria, of which the default value is' ARC '-- adjust Rand index.
        best : boolean, optional
                Whether to find the best model, the default value is True.

        return value
        -------
        nothing

        """
     #The dictionary is converted to a dataframe and the dictionary length check is performed
        try:
            df_dict = pd.DataFrame(param_dict).T
        except:
            print("the length of each key in the hyperparameter dictionary is inconsistent, please check")
        
        # Initialization (prevent the internal parameter from increasing infinitely)
        self.__init__()
        
        # initial assignment
        dbscan = clone(self.model)
        self.score_measure = measure
        
        # Other class statistics initialization
        ndratio_dict = {'noise_data_ratio':[]}
        self.otherstats.update(ndratio_dict)
        
        # According to the grid hyperparameters, modify the corresponding parameters of dbscan object, train the model, and update the internal parameter.
        for col in range(len(df_dict.columns)):
            for row in range(len(df_dict.index)):
                param_name = df_dict[col].index[row]
                param_value = df_dict[col][row]
                setattr(dbscan,param_name,param_value)
        
            # train model
            model = clone(dbscan)
            model.fit(data)
            
            # Count the number of clusters (without noise) 
            clusters = set(model.labels_)
            clusters.remove(-1)           
            self.clusters.append(len(clusters))
             
            # Count the number of samples in each cluster (without noise), and cluster noise 
            label_sr = pd.Series(model.labels_)
            self.piec.append(label_sr[label_sr != -1].value_counts().values)
            self.outliers.append(label_sr[label_sr == -1].value_counts().values)
             
             # other statistical values
             # noise ratio
            ndratio = sum(model.labels_ == -1) / len(model.labels_)
            self.otherstats.get('noise_data_ratio').append(ndratio)
             
            # Storage model prediction label
            self.pred_labels.append(model.labels_)
             
            
        #If  best == True, the scores of all models are calculated according to the evaluation criteria.
        if best == True:
            self.score_measure = measure
            for pred_labels in self.pred_labels:
                score = get_measure_scores(measure, data, pred_labels, labels) # Return the evaluation value of the model
                self.scores.append(score)
               
      # find the best model
        best_index = get_best_score_index(self.scores, measure)   # return the index of the best model
        self.best_model_param = pd.DataFrame(df_dict.T.iloc[best_index]).T
        self.best_score = self.scores[best_index]
        self.best_index = best_index        
        
        # output information
        print("model training completed")
        print("The number of total training models{}".format(len(df_dict.columns)))
              # output information
        print("model training completed")
        print("The number of total training models{}".format(len(param_grid)))

    # Save the best model information into the output file
    def best_model_to_csv(self,outputfile):
        
        df = self.best_model_param.copy()
        df['measure'] = self.score_measure
        df['score'] = self.best_score
        
        df.to_csv(outputfile)
        print('The results have been saved to the file{}'.format(outputfile))
            


# Get input hyperparameters
def get_hyper_variable_dict(dataframe):
    """
    Get the hyperparameter dictionary according to the dataframe

    Input parameters
    ----------
    dataframe : pandas dataframe
        dataframe with hyperparameters.

    return value
    -------
    hyperparameter dictionary

    """
    
    # The input dataframe is processed and transformed into a dictionary
    df = dataframe.T
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    var_dict = df.to_dict('list')
    
    # Remove NAN from dictionary
    for key, values in var_dict.items():
        var_dict[key] = []
        for value in values:
            if  pd.notnull(value):
                var_dict[key].append(value)
    
    return(var_dict)
 
# Get control parameters
def get_control_var(dataframe):
    """
    Get program control parameter list according to dataframe

    Input parameters
    ----------
    dataframe : pandas dataframe
        dataframe with hyperparameters.

    return value
    -------
    control parameters list

    """
    
    control_list = []
    df = dataframe
    mode = df[df['VarName'] == 'Mode']['Value'][0]
        
    # control parameters list
    control_list.append(mode)
    
    return(control_list)
