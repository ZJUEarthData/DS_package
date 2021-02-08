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
        col: A list of columns that need to plot
        df: The dataframe 
    Output:
        A heatmap describing the correlation between the required columns
    '''
    plot_df = df[col]
    plot_df_cor = plot_df.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(plot_df_cor, cmap = 'coolwarm', annot=True, linewidths=.5)



def distribution_plot(col, df):
    '''
    Input: 
        col: A list of columns that need to plot
        df: The dataframe 
    Output:
        A large graph containing the respective distribution subplots of the required columns
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
        col: A list of columns that need to plot
        df: The dataframe 
    Output:
        A large graph containing the respective distribution subplots after log transformation of the required columns
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
        col: A list of columns that need to plot
        df_origin: The original dataframe 
        df_impute: The dataframe after missing value imputation
    Output:
        A large graph containing the respective probability plots (origin vs. impute) of the required columns
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












