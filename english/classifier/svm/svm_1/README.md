Here we use SVM_ cloud.py to explain. (other codes are used in a similar way)
Since the code is used in a similar way, we will only talk about “svm.cloud.py”.

 
1.Basic Ideas
The concrete idea of the code is as follows: input paths, input data and parameter tables, build models, evaluate different models, grid search for optimal models, train models, save models and parameters, draw learning curves.


2.Function Declaration

（1）SVMGridsearch
Constructing models with different parameter combinations
dmodel: 		The default model
data: 		Training set
labels: 		Labels of training set
param_dict: 	The table of hyper-parameter

（2）get_marks
  evaluation
estimator:                The model you want to evaluate
data:                        Data set
labels:		Labels of data set
name:		Name of model

（3）read_para
  read hyper-parameter table
FEATURE_FILE_PATH:	the path of hyper-parameter table

（4）plot_learning_curve
       draw learning curve
model:		The model you want to evaluate
data:	                Data set
labels:		Labels of data set
OUTPUT_RESULT:	The path to output result

（5）write_wrong_label
        output misclassified data
classifier: 	                model
data_list:		data set
label_list:		labels
file_name:	output path

3. Parameter Specification
FEATURE_FILE_PATH: 	The path of hyper-parameter table
DATA_FILE_PATH:		The path of data
OUTPUT_MODEL:		The path to output model
OUTPUT_RESULTS: 	The path to output result

df:		data set
data:		data set without labels
labels:	                Labels

model:			Original model
svm_dict:			Parameter table
output:		                Models with different parameter combinations
svm_best_model:	                best model of grid search
svm_result:		The result after fit the data

result:		A table that holds the best parameters




