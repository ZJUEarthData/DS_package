#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:38:39 2020

@author: Dan Hu
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, kruskal
import random

def test_once(df_orig, df_impute, test = 'wilcoxon'):
    '''
    Input:
        df_orig: The original dataset with missing value
        df_impute: The dataset after imputation
        test: The statistics test used    
    Output:
        A numpy array containing the p-values of the tests on each column in the column order
    '''   
    cols = df_orig.columns
    pvals = np.array([])
    
    if test == 'wilcoxon':
        for c in cols:
            try:
                stat, pval = wilcoxon(df_orig[c], df_impute[c])
                pvals = np.append(pvals, pval)
            except:
                pvals = np.append(pvals, 0)
                
    if test == 'kruskal':
        for c in cols:
            stat, pval = kruskal(df_orig[c], df_impute[c], nan_policy = 'omit')
            pvals = np.append(pvals, pval)
            
    return pvals


def monte_carlo_simulator(df_orig, df_impute, sample_size, iteration, test = 'wilcoxon', confidence = 0.05):
    '''
    Input:
        df_orig: The original dataset with missing value
        df_impute: The dataset after imputation
        test: The statistics test used
        sample_size: The size of the sample for each iteration
        iteration: Number of iterations of Monte Carlo Simulation
        confidence: Confidence level, default to be 0.05
    Output:
        The column names that reject the null hypothesis
    '''
    random.seed(2)
    simu_pvals = np.array([0] * df_orig.shape[1])
    for i in range(iteration):
        sample_idx = random.sample(range(df_orig.shape[0]), sample_size)
        sample_orig = df_orig.iloc[sample_idx]
        sample_impute = df_impute.iloc[sample_idx]
        
        one_pval = test_once(df_orig = sample_orig, df_impute = sample_impute, test = test)
        simu_pvals = simu_pvals + one_pval
    
    col_res = simu_pvals / iteration
    return df_orig.columns[np.where(col_res < confidence)[0]]









