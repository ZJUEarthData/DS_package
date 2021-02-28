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
        df_orig: 缺少值的原始数据集
        df_impute: 输入后的数据集
        test: 使用统计的方法测试   
    Output:
        一个numpy数组，按列的顺序包含每个列上测试的p值
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
        df_orig: 缺少值的原始数据集
        df_impute: 输入后的数据集
        test: 使用统计的方法进行测试
        sample_size: 每次迭代的样本大小
        iteration: 蒙特卡罗模拟的迭代次数
        confidence: 置信度级别，默认为0.05
    Output:
        拒绝零假设的列名
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









