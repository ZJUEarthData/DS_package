# -*- coding: utf-8 -*-
"""lofplus.ipynb

A python module to perform Local Outlier Factor analysis for Outlier identification and subsequent plotting.
Dependencies: NumPy, Pandas, Plotly Express, Matplotlib.Pyplot, SKLearn, Math

"""

def random_subspace(features,subspaces, min_features=None, max_features=None):
  '''
  To generate a list of subsets of all features
  
  Parameters:
  features (list): A list of features.
  subspaces (int): An integer denoting the number of non-unique subspaces to be created.
  min_features (int): Minimum number of features to be used to create subspaces. (default= None. half the total number of features.)
  max_features (int): Maximum number of features to be used to create subspaces. (default= None. the total number of features.)

  Returns:
  list: A nested list of features.
  '''
  import numpy as np
  import pandas as pd
  from sklearn.neighbors import LocalOutlierFactor
  import random
  import math

  if min_features==None: min_features=math.floor(len(features)/2)
  if max_features==None: max_features=len(features)
  if max_features>len(features): max_features=len(features)

  feature_list=[]
  for i in range(subspaces):  
    no_features=np.random.randint(low=min_features,high=max_features)
    rand_features=random.sample(population=features,k=no_features)
    feature_list.append(rand_features)
  return feature_list

def detect(data, no_of_subspaces,
           tot_feature,n_neighbors=50, contamination="auto",
           min_features=2, max_features=None,
           separate_df=False):
  '''
  To perform Local Outlier Factor outlier analysis. Adds a 'label' column to the passed dataframe having the analysis result.
  
  Parameters:
  data (DataFrame): A DataFrame.
  no_of_subspaces (int): An integer denoting the number of non-unique subspaces to be created.
  tot_feature (list): A list of all features to be used for analysis.
  n_neighbors (int): Number of neighbours. (default= 50)
  contamination (float): Number between 0 and 0.5 denoting proportion of outliers. (default= 'auto')
  min_features (int): Minimum number of features to be used to create subspaces. (default= None. uses half the total number of features.)
  max_features (int): Maximum number of features to be used to create subspaces. (default= None. uses the total number of features.)
  separate_df(boolean): Whether to create separate dataframes for outliers and inliers.

  Returns:
  tuple: A tuple of two dataframes outliers and inliers respectively in the format (outliers,inliers) if separate_df=True.
  '''
  import numpy as np
  import pandas as pd
  from sklearn.neighbors import LocalOutlierFactor
  import random
  import math
  
  df=data[tot_feature].dropna()
  feature_list=random_subspace(tot_feature, no_of_subspaces,min_features, max_features)
  outlier_labels=pd.DataFrame(index=df.index)

  model=LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,n_jobs=-1)

  for i in range(no_of_subspaces):
    df_temp=df[feature_list[i]]
    y_pred=model.fit_predict(df_temp)
    outlier_labels[str("Model "+str(i+1))]=pd.DataFrame(y_pred, index=df.index)
  
  outlier_labels["Total"]=outlier_labels.sum(axis=1)
  labels=[]
  for i in outlier_labels["Total"]:
    if i <0: labels.append("Outlier")
    else: labels.append("Inlier")
 
  df['label']= pd.DataFrame(labels, index=df.index)
  data['label']=df['label']
  data['label']=data['label'].replace(np.nan,"Undetermined")

  if separate_df:
    outlier_df=df.loc[df[df["label"]=="Outlier"].index].drop(['label'],axis=1)
    inlier_df=df.loc[df[df["label"]=="Inlier"].index].drop(['label'],axis=1)
    return  (outlier_df,inlier_df)

def plot(data,features,title="LOF Plot",col_name='label',save_fig=True):
  '''
  To generate a 2D/3D Plotly express plot.
  
  Parameters:
  data (DataFrame): A DataFrame.
  features (list): A list of length 2 or 3 or less of corresponding plot axes.
  title (string): Plot name. (default: 'LOF Plot')
  col_name (string): Name of column in the dataframe having the outlier labels. (default: 'label')
  save_fig (boolean): whether to save the figure as an external HTML file. (default: True)

  Returns:
  fig: A 3D plotly scatter plot
  html: 3D plotly scatter plot file
  '''
  import plotly.express as px
  import matplotlib.pyplot as plt

  if len(features) not in [2,3]:
    import sys
    sys.exit("Wong length of features")
  
  elif len(features)==3:
    fig = px.scatter_3d(data, x=features[0], y=features[1], z=features[2],
                      color=col_name,labels={'Inlier':"Inlier", 'Outlier':"Outlier", 'Undetermined':"Undetermined"},color_discrete_map={'Inlier':"#1F77B4", 'Outlier':"orangered", 'Undetermined':"silver"},
                    title=title)
  elif len(features)==2:
      fig = px.scatter(data, x=features[0], y=features[1],
                      color=col_name,labels={'Inlier':"Inlier", 'Outlier':"Outlier", 'Undetermined':"Undetermined"},color_discrete_map={'Inlier':"#1F77B4", 'Outlier':"orangered", 'Undetermined':"silver"},
                    title=title)  
  
  fig.update_layout(template='plotly_dark')
  fig.show()
  if save_fig:
    fig.write_html(str(str(title)+".html"))