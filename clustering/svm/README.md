这里以svm_cloud.py进行说明（其他代码的使用方法类似）
Since the code is used in a similar way, we will only talk about “svm.cloud.py”.

 
一、基本思路（Basic Ideas）
svm算法设计思路如下：读入路径，读入数据和参数表，构建模型，测试不同模型的评分，网格搜索最优模型，训练模型，保存模型和参数，绘制学习曲线。
The concrete idea of the code is as follows: input paths, input data and parameter tables, build models, evaluate different models, grid search for optimal models, train models, save models and parameters, draw learning curves.


二、函数说明（function declaration）
（1）SVMGridsearch
构建不同参数组合的模型 (Constructing models with different parameter combinations)
dmodel: 		默认模型 (The default model)
data: 		训练集 (Training set)
labels: 		训练集标签 (Labels of training set)
param_dict: 	超参数表 (The table of hyper-parameter)

（2）get_marks
获取评分(evaluation)
estimator:		需要获取评分的模型 (The model you want to evaluate)
data:			数据集 (Data set)
labels:		数据集标签 (Labels of data set)
name:		模型方法名 (name of model)

（3）read_para
读取超参数表(read hyper-parameter table)
FEATURE_FILE_PATH:	参数表路径(the path of hyper-parameter table)

（4）plot_learning_curve
绘制学习曲线(draw learning curve)
model:		需要绘制曲线的模型(The model you want to evaluate)
data:			数据集(Data set)
labels:		数据集标签(Labels of data set)
OUTPUT_RESULT:	输出图像路径 (The path to output result)

（5）write_wrong_label
输出错误分类数据（output misclassified data）
classifier: 	模型（model）
data_list:		数据集(data set)
label_list:		标签(labels)
file_name:	输出路径(output path)
三、参数说明（parameter specification）
FEATURE_FILE_PATH	: 	参数表路径(the path of hyper-parameter table)
DATA_FILE_PATH:		数据路径(The path of data)
OUTPUT_MODEL:		模型输出路径(The path to output model)
OUTPUT_RESULTS: 		结果输出路径(The path to output result)

df:		读入的数据集(data set)
data:		去除标签的数据集(data set without labels)
labels:	标签(Labels)

model:			初始模型(model)
svm_dict:			读取的超参数表(hyper-parameter table)
output:		根据超参数表生成的模型(models with different parameter combinations)
svm_best_model:	网格搜索最优模型(best model)
svm_result:		模型训练后的结果(the result after fit the data)

result:		保存着最优参数的表(A table that holds the best parameters)




