# -*- coding: utf-8 -*-
"""
Created on 17/08/2020

@author: Jingwei Liu

Version 1.1
"""


from sklearn import metrics

# 返回模型评价值
def get_measure_scores(measure, data, pred_labels, real_labels = None):
    """获取对应评价标准的评价分数
    评价方法共2种： （未来会添加更多方法）
        1. ARC : 调整兰德指数 (Adjusted Rand Score)
        2. AMIC : 调整互信息(Adjusted Mutual Information Score)
        3. V : 同质性与完整性的调和平均数（V Measure Score）
        4. Homegeneity : 同质性
        5. Completeness : 完整性
        6. Silouette : 轮廓系数
        7. CHS : Calinski Harabasz Score

    输入参数
    ----------
    measure : string
        评价标准.
    data : pandas dataframe
        要评价的数据.
    pred_labels : array-like of shape
        预测的数据标签.
    real_labels : array-like of shape, optional
        真实的数据标签，默认值为None.

    错误抛出
    ------
    ValueError
       评价标准字符串输入错误.

    返回值
    -------
    评价值

    """
    
    if measure == 'ARC':
        score = _get_Adjust_rand_score(real_labels, pred_labels)
    elif measure == 'AMIC':
        score = _get_adjusted_mutual_info_score(real_labels, pred_labels)
    elif measure == 'V':
        score = _get_v_measure_score(real_labels, pred_labels)
    elif measure == 'Homegeneity':
        score = _get_homogeneity_score(real_labels, pred_labels)
    elif measure == 'Completeness':
        score = _get_completeness_score(real_labels, pred_labels)
    elif measure == 'Silouette':
        score = _get_silhouette_score(data, pred_labels)
    elif measure == 'CHS':
        score = _get_calinski_harabasz_score(data, pred_labels) 
    else:
        raise ValueError("请选择正确的评价标准")
    

    return(score)

# 显示模型的所有评价值
def get_marks(data, pred_labels, real_labels):
    """显示模型所有的评价值
    

    输入参数
    ----------
    data : pandas dataframe
        要评价的数据.
    pred_labels : array-like of shape
        预测的数据标签.
    real_labels : array-like of shape, optional
        真实的数据标签，默认值为None.

    返回值
    -------
    无.

    """
    
    print("Adjusted Rand Score:{}",format(_get_Adjust_rand_score(real_labels, pred_labels)))
    print("Adjusted Mutual Info Score:{}",format(_get_adjusted_mutual_info_score(real_labels, pred_labels)))
    print("V Measure Score:{}",format(_get_v_measure_score(real_labels, pred_labels)))
    print("Homogeneity Score:{}",format(_get_homogeneity_score(real_labels, pred_labels)))
    print("Completeness Score:{}",format(_get_completeness_score(real_labels, pred_labels)))
    print("Silhouette Score:{}",format(_get_silhouette_score(data, pred_labels)))
    print("Calinski Harabasz Score:{}",format(_get_calinski_harabasz_score(data, pred_labels) ))

        
# 找到最佳评分的位置索引
def get_best_score_index(score_list, measure):
    """返回最好评价值得位置索引
    
    输入参数
    ----------
    score_list : list
        评分的列表（list）
    measure : string
        评价标准，

    返回值
    -------
    best_index: int
        位置索引

    """
    
    if measure == 'ARC':
        best_index = score_list.index(max(score_list))
    elif measure == 'AMIC':
        best_index = score_list.index(max(score_list))
    elif measure == 'V':
        best_index = score_list.index(max(score_list))
    elif measure == 'Homegeineity':
        best_index = score_list.index(max(score_list))
    elif measure == 'Completeness':
        best_index = score_list.index(max(score_list))
    elif measure == 'Silouette':
        best_index = score_list.index(max(score_list))
    elif measure == 'CHS':
        best_index = score_list.index(max(score_list))
    else:
        raise ValueError("请选择正确的评价标准")
    
    return(best_index)


# 调整兰德指数
def _get_Adjust_rand_score(real_labels, pred_labels):
    """返回调整兰德指数
    
    输入参数
    --------
    real_labels : array-like of shape
        真实的数据标签.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    调整兰德指数
    
    """
    
    return(metrics.adjusted_rand_score(real_labels, pred_labels))
    
# 调整互信息
def _get_adjusted_mutual_info_score(real_labels, pred_labels):
    """返回调整互信息
    
    输入参数
    --------
    real_labels : array-like of shape
        真实的数据标签.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    调整互信息
    
    """    
    
    return(metrics.adjusted_mutual_info_score(real_labels, pred_labels))

# V Measure Score
def _get_v_measure_score(real_labels, pred_labels):
    """返回 V Measure Score
    
    输入参数
    --------
    real_labels : array-like of shape
        真实的数据标签.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    V Measure Score
    
    """  
     
    return(metrics.v_measure_score(real_labels, pred_labels))

# 同质性
def _get_homogeneity_score(real_labels, pred_labels):
    """返回同质性
    
    输入参数
    --------
    real_labels : array-like of shape
        真实的数据标签.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    同质性
    
    """  
    
    return(metrics.homogeneity_score(real_labels, pred_labels))

# 完整性
def _get_completeness_score(real_labels, pred_labels):
    """返回完整性
    
    输入参数
    --------
    real_labels : array-like of shape
        真实的数据标签.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    完整性
    
    """  
    
    return(metrics.completeness_score(real_labels, pred_labels))

# 轮廓系数
def _get_silhouette_score(data, pred_labels):
    """返回轮廓系数
    
    输入参数
    --------
    data : pandas dataframe
        要评价的数据.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    轮廓系数
    
    """  
    
    return(metrics.silhouette_score(data, pred_labels))

# Calinski Harabasz Score
def _get_calinski_harabasz_score(data, pred_labels):
    """返回Calinski Harabasz Score
    
    输入参数
    --------
    data : pandas dataframe
        要评价的数据.
    pred_labels : array-like of shape
        预测的数据标签.
    
    返回值
    --------
    Calinski Harabasz Score
    
    """  
    
    return(metrics.calinski_harabasz_score(data, pred_labels))
