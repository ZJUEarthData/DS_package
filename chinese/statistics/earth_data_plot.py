#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:41:45 2020

@author: Dan Hu
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm



def correlation_plot(col, df):
    '''
    Input: 
        col: 一个需要绘图（plot）的列（comumns）
        df: 数据结构（dataframe）
    Output:
        一个描绘所给列关系之间的热图（heatmap）
    '''
    plot_df = df[col]
    plot_df_cor = plot_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(plot_df_cor, cmap = 'coolwarm', annot=True, linewidths=.5)



def distribution_plot(col, df):
    '''
    Input: 
        col: 一个需要绘图（plot）的列（comumns）
        df: 数据结构（dataframe） 
    Output:
        包含所需列的各自分布子图的大型图表
    '''
    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n*2, n*2))
    for i in range(len(col)):
        plt.subplot(n, n, i+1)
        plt.hist(df[col[i]])
        plt.title(col[i])
    plt.tight_layout()


def logged_distribution_plot(col, df):
    '''
    Input: 
        col: 一个需要绘图（plot）的列（comumns）
        df: 数据结构（dataframe） 
    Output:
       在对所需列进行对数变换后，包含各自分布子图的大型图表
    '''
    n = int(np.sqrt(len(col))) + 1
    plt.figure(figsize=(n*2, n*2))
    for i in range(len(col)):
        plt.subplot(n, n, i+1)
        plt.hist(df[col[i]].map(lambda x: np.log(x+1)))
        plt.title(col[i])
    plt.tight_layout()
    
    
def probability_plot(col, df_origin, df_impute):
    '''
    Input: 
        col: 一个需要绘图（plot）的列（comumns）
        df_origin: 原始的数据结构
        df_impute: 缺失值赋值后的数据结构
    Output:
        一个包含所需列的各自概率图(原点与归因)的大图表
    '''

    r, c = len(col) // 4 + 1, 4
    fig = plt.figure(figsize=(c*8, r*8))
    for i in range(len(col)):   
        feature = col[i]
        pp_origin = sm.ProbPlot(df_origin[feature].dropna(), fit=True)
        pp_impute = sm.ProbPlot(df_impute[feature], fit=True)
        ax = fig.add_subplot(r, c, i+1)
        pp_origin.ppplot(line="45", other=pp_impute, ax=ax)
        plt.title(f"{feature}, origin vs. impute")

    plt.tight_layout()    












