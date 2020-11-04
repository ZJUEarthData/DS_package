# -*- coding: utf-8 -*-
"""
Created on 17/08/2020

@author: Jingwei Liu

Version 1.1
"""


from sklearn import metrics

# Return model evaluation value
def get_measure_scores(measure, data, pred_labels, real_labels = None):
    """
    Get the evaluation score of the corresponding evaluation standard
    There are two evaluation methods: (more methods will be added in the future)
        1. ARC : Adjusted Rand Score
        2. AMIC : Adjusted Mutual Information Score
        3. V : the harmonic mean of homogeneity and completeness （V Measure Score）
        4. Homogeneity 
        5. Completeness 
        6. Silouette : Silhouette Coefficient
        7. CHS : Calinski Harabasz Score

    Input parameters
    ----------
    measure : string
        evaluation criteria.
    data : pandas dataframe
        Data to be evaluated.
    pred_labels : array-like of shape
        Forecast data label.
    real_labels : array-like of shape, optional
         Real data label, the default value is None.

    错误抛出
    ------
    ValueError
       Evaluation criteria string input error.

    Return value
    -------
    Evaluation value

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
        raise ValueError("Please select the correct evaluation criteria.")
    

    return(score)

# Display all evaluation values of the model
def get_marks(data, pred_labels, real_labels):
    """Display all evaluation values of the model
    

    Input parameters
    ----------
    data : pandas dataframe
       Data to be evaluated.
    pred_labels : array-like of shape
        Forecast data label.
    real_labels : array-like of shape, optional
         Real data label, the default value is None.

    Return value
    -------
    None.

    """
    
    print("Adjusted Rand Score:{}",format(_get_Adjust_rand_score(real_labels, pred_labels)))
    print("Adjusted Mutual Info Score:{}",format(_get_adjusted_mutual_info_score(real_labels, pred_labels)))
    print("V Measure Score:{}",format(_get_v_measure_score(real_labels, pred_labels)))
    print("Homogeneity Score:{}",format(_get_homogeneity_score(real_labels, pred_labels)))
    print("Completeness Score:{}",format(_get_completeness_score(real_labels, pred_labels)))
    print("Silhouette Score:{}",format(_get_silhouette_score(data, pred_labels)))
    print("Calinski Harabasz Score:{}",format(_get_calinski_harabasz_score(data, pred_labels) ))

        
# Find the location index of the best score
def get_best_score_index(score_list, measure):
    """Returns the location index of the best evaluation value
    
    Input parameters
    ----------
    score_list : list
        Score list（list）
    measure : string
        Evaluation criteria，

    Return value
    -------
    best_index: int
        location index

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
        raise ValueError("Please select the correct evaluation criteria.")
    
    return(best_index)


# Adjusted Rand Score
def _get_Adjust_rand_score(real_labels, pred_labels):
    """Return Adjusted Rand Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    Adjusted Rand Score
    
    """
    
    return(metrics.adjusted_rand_score(real_labels, pred_labels))
    
# Adjust Mutual Information
def _get_adjusted_mutual_info_score(real_labels, pred_labels):
    """Return adjust mutual information
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    Adjust Mutual Information
    
    """    
    
    return(metrics.adjusted_mutual_info_score(real_labels, pred_labels))

# V Measure Score
def _get_v_measure_score(real_labels, pred_labels):
    """Return V Measure Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    V Measure Score
    
    """  
     
    return(metrics.v_measure_score(real_labels, pred_labels))

# homogeneity
def _get_homogeneity_score(real_labels, pred_labels):
    """Return Homogeneity Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    homogeneity
    
    """  
    
    return(metrics.homogeneity_score(real_labels, pred_labels))

# completeness
def _get_completeness_score(real_labels, pred_labels):
    """Return Completeness Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    completeness
    
    """  
    
    return(metrics.completeness_score(real_labels, pred_labels))

# Silhouette 
def _get_silhouette_score(data, pred_labels):
    """Return Silhouette Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    Silhouette 
    
    """  
    
    return(metrics.silhouette_score(data, pred_labels))

# Calinski Harabasz Score
def _get_calinski_harabasz_score(data, pred_labels):
    """Return Calinski Harabasz Score
    
    Input parameters
    --------
    real_labels : array-like of shape
        Real data label.
    pred_labels : array-like of shape
        Forecast data label.
    
    Return value
    --------
    Calinski Harabasz Score
    
    """  
    
    return(metrics.calinski_harabasz_score(data, pred_labels))
