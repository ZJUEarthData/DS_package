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
        
    # 初始化
    def __init__(self):
        """
        初始化类

        内部参数
        --------
        model : DBSCAN的默认模型
        clusters : 用来存储训练所有模型的簇个数
        piec : 用来存储所有训练模型每个簇的样本数
        otherstates : 其他统计数据，例如噪声比
        outliers : 用来存储所有训练模型噪点个数
        pred_labels : 用来储存所有训练模型对训练数据的预测标签
        score_measure : 用来选择最好模型的评价标准（一般为首字母缩写） 例如：调整兰德指数： ARC
        scores : 用来储存所有模型的评价分数
        best_model_param : 所有模型中最好的模型的超参数
        best_score : 最好模型的评价分数
        best_index : 最好模型的索引

        """
            
        self.model = DBSCAN()
        self.clusters = []
        self.piec = []  # points in each cluster 每个簇的样本数
        self.otherstats = {}
        self.outliers = []
        self.pred_labels = []
        self.score_measure = None
        self.scores = []
        self.best_model_param = {}
        self.best_score = None
        self.best_index = None

    # 构建网格搜索来获取DBSCAN算法在不同超参数下的效果
    # 字典中各key的数据长度可以不一样
    def DBSCANGridsearch(self, data, param_dict, labels = None, measure = 'ARC', best = True):
        """
        DBSCAN 的网格搜索

        输入参数
        ----------
        data : pandas dataframe
            训练数据（不含真实标签）.
        param_dict : dictionary
            含有超参数的字典，用于构建搜索网格，字典中各key的数据长度可以不一样.
        labels : array-like of shape, optional
            真实数据标签，默认值为 None.
        measure : string, optional
            评价标准, 默认值为 'ARC' -- 调整兰德指数.
        best : boolean, optional
            是否寻找最好模型， 默认值为 True.

        返回值
        -------
        无

        """
        
        # 初始化（防止内部参数无限增大）
        self.__init__()
        
        # 初始赋值
        dbscan = clone(self.model)
        self.score_measure = measure
        
        # 其他类统计值初始化
        ndratio_dict = {'noise_data_ratio':[]}
        self.otherstats.update(ndratio_dict)  
           
        # 构建超参数网格
        param_grid = ParameterGrid(param_dict)
         
        # 依据网格超参数修改dbscan object 的对应参数，训练模型，更新内部参数  
        for param in param_grid:
            for key, value in param.items():
                setattr(dbscan,key,value)
       
            # 训练模型
            model = clone(dbscan)
            model.fit(data)
            
            # 统计簇个数（不含噪点）
            clusters = set(model.labels_)
            clusters.remove(-1)           
            self.clusters.append(len(clusters))
             
            # 统计每个簇样本数（不含噪点）,以及聚类噪点数
            label_sr = pd.Series(model.labels_)
            self.piec.append(label_sr[label_sr != -1].value_counts().values)
            self.outliers.append(label_sr[label_sr == -1].value_counts().values)
             
            # 其他统计值
            # 噪声比
            ndratio = sum(model.labels_ == -1) / len(model.labels_)
            self.otherstats.get('noise_data_ratio').append(ndratio)
             
            # 存模型预测标签
            self.pred_labels.append(model.labels_)
             
            
        # 如果 best == True, 根据评价标准，求出所有模型的分数
        if best == True:
            self.score_measure = measure
            for pred_labels in self.pred_labels:
                score = get_measure_scores(measure, data, pred_labels, labels)  # 返回模型的评价值
                self.scores.append(score)
               
        # 找到最佳模型
        best_index = get_best_score_index(self.scores, measure)  # 返回最佳模型的索引
        self.best_model_param = pd.DataFrame(param_grid[best_index],index=[0])
        self.best_score = self.scores[best_index]
        self.best_index = best_index        
        
        #输出信息
        print("模型训练完成")
        print("共训练模型{}个".format(len(param_grid)))
  
    
    # 批处理来获取DBSCAN算法在不同超参数下的效果
    # 字典中各key的数据长度必须一样
    def DBSCANBatchSearch(self, data, param_dict, labels = None, measure = 'ARC', best = True):
        """
        DBSCAN 批处理搜索

        输入参数
        ----------
        data : pandas dataframe
            训练数据（不含真实标签）.
        param_dict : dictionary
            含有超参数的字典，用于构建搜索网格，字典中各key的数据长度必须一样.
        labels : array-like of shape, optional
            真实数据标签，默认值为 None.
        measure : string, optional
            评价标准, 默认值为 'ARC' -- 调整兰德指数.
        best : boolean, optional
            是否寻找最好模型， 默认值为 True.

        返回值
        -------
        无

        """
        # 转换字典为dataframe，同时执行字典长度检查
        try:
            df_dict = pd.DataFrame(param_dict).T
        except:
            print("超参数字典各key长度不一致，请检查")
        
        # 初始化（防止内部参数无限增大）
        self.__init__()
        
        # 初始赋值
        dbscan = clone(self.model)
        self.score_measure = measure
        
        # 其他类统计值初始化
        ndratio_dict = {'noise_data_ratio':[]}
        self.otherstats.update(ndratio_dict)
        
        #依据字典超参数修改dbscan object 的对应参数，训练模型，更新内部参数  
        for col in range(len(df_dict.columns)):
            for row in range(len(df_dict.index)):
                param_name = df_dict[col].index[row]
                param_value = df_dict[col][row]
                setattr(dbscan,param_name,param_value)
        
            # 训练模型
            model = clone(dbscan)
            model.fit(data)
            
            # 统计簇个数（不含噪点）
            clusters = set(model.labels_)
            clusters.remove(-1)           
            self.clusters.append(len(clusters))
             
            # 统计每个簇样本数（不含噪点）,以及聚类噪点数
            label_sr = pd.Series(model.labels_)
            self.piec.append(label_sr[label_sr != -1].value_counts().values)
            self.outliers.append(label_sr[label_sr == -1].value_counts().values)
             
            # 其他统计值
            # 噪声比
            ndratio = sum(model.labels_ == -1) / len(model.labels_)
            self.otherstats.get('noise_data_ratio').append(ndratio)
             
            # 存模型预测标签
            self.pred_labels.append(model.labels_)
             
            
        # 如果 best == True, 根据评价标准，求出所有模型的分数
        if best == True:
            self.score_measure = measure
            for pred_labels in self.pred_labels:
                score = get_measure_scores(measure, data, pred_labels, labels)  # 返回模型的评价值
                self.scores.append(score)
               
        # 找到最佳模型
        best_index = get_best_score_index(self.scores, measure)  # 返回最佳模型的索引
        self.best_model_param = pd.DataFrame(df_dict.T.iloc[best_index]).T
        self.best_score = self.scores[best_index]
        self.best_index = best_index        
        
        # 输出信息
        print("模型训练完成")
        print("共训练模型{}个".format(len(df_dict.columns)))

    # 把最佳模型信息存入输出文件
    def best_model_to_csv(self,outputfile):
        
        df = self.best_model_param.copy()
        df['measure'] = self.score_measure
        df['score'] = self.best_score
        
        df.to_csv(outputfile)
        print('已将结果存入文件{}'.format(outputfile))
            


# 获取取输入超参数参数
def get_hyper_variable_dict(dataframe):
    """
    根据dataframe获取超参数字典

    输入参数
    ----------
    dataframe : pandas dataframe
        含有超参数的dataframe.

    返回值
    -------
    超参数字典.

    """
    
    # 处理输入的dataframe，将其变换为字典
    df = dataframe.T
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    var_dict = df.to_dict('list')
    
    # 去除字典中的NAN 
    for key, values in var_dict.items():
        var_dict[key] = []
        for value in values:
            if  pd.notnull(value):
                var_dict[key].append(value)
    
    return(var_dict)
 
# 获取控制参数
def get_control_var(dataframe):
    """
    根据dataframe获取程序控制参数list

    输入参数
    ----------
    dataframe : pandas dataframe
        含有控制参数的dataframe.

    返回值
    -------
    控制参数list

    """
    
    control_list = []
    df = dataframe
    mode = df[df['VarName'] == 'Mode']['Value'][0]
        
    # 控制参数list
    control_list.append(mode)
    
    return(control_list)
