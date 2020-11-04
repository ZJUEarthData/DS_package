# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 01:04:10 2020

@author: Jingwei Liu

Version 1.1
"""

import sys
sys.path.append("C:/Users/Jingwei Liu/OneDrive - Auburn University/Geo_Cooperation/Old Algorithm Encapsulation")
import pandas as pd
import numpy as np
from GeoDBSCAN import *


if __name__=='__main__':

    # Check system input parameters
    if (len(sys.argv) != 4):
        raise Exception('Input parameters should be 3, please re-enter parameters')
        
    # Get system input parameters
    data_file = sys.argv[1]
    vars_file = sys.argv[2]
    output_file = sys.argv[3]
    
    
    # get file data, hyperparameters, control parameters
    df = pd.read_excel(data_file)
    df_hypervar = pd.read_excel(vars_file, sheet_name = "DBSCAN_HyperVars")
    df_controlvar = pd.read_excel(vars_file, sheet_name = "DBSCAN_Control")
    
    # training data and label separation
    data = df.drop('TRUE VALUE', axis = 1)   
    labels = df['TRUE VALUE']
    
    
    dbscan = GeoDBSCAN()
    dbscan_dict = get_hyper_variable_dict(df_hypervar)
    
    control = get_control_var(df_controlvar)
    
    if control[0] == 'GridSearch':    
        dbscan.DBSCANGridsearch(data,dbscan_dict,labels)
    elif control[0] == 'BatchSearch':
        dbscan.DBSCANBatchSearch(data,dbscan_dict,labels)
        
    
    # Output files
    dbscan.best_model_to_csv(output_file)

