Instructions for Clustering Algorithm

1.Brief Introduction
All the clustering algorithms in the package use GridSearchCV to find optimal parameters，train the model and saving the results. 
There are four algorithms, and every algorithm has three files(XX.py, XX_class.py, XX_cloud.py) that can run in different ways, and each of these files requires a data set and a hyper-parameter table as input.
They can be selected according to the needs.(XX refers to one of AP, kmeans, Meanshift, svm)


2.File description
Data set: 
An excel sheet used to store training data., the file “test4.xlsx” is a sample.


Hyper-parameter table:
An excel sheet, it contains the value of the hyper-parameter you want to test. The leftmost column is the hyper-parameter name, on the right is the value of the parameter you want to test. The file “para.xlsx” is a sample.


XX.py: 
This file can be run directly through the terminal, but make sure the name of the data set and the hyper-parameter table match their names in the code.

XX_class.py:
This file puts all the functions used by XX.py in one class. You can import this class  in another Python file when you want to use it.



XX_cloud.py:
This file can be run directly through the terminal using the following format:
python XX_cloud.py [parameter_file_path] [data_file_path] [output_model_path] [output_result_path]
The file supports external input path, and the format is as above in the terminal. Input the path of hyperparameter table, data set, model output and result output. Here is an example of using the AP algorithm.
Sample:
python AP_cloud.py ./para.xlsx ./test4.xlsx ./ ./
( ‘./’ means current folder)


test.pkl:
The best model after training will be saved in this file. XX_predict_cloud.py will load this model to predict the test set.


XX_predict_cloud.py:
Loading model and making prediction. You should use the following format:
python AP_predict_cloud.py [data_file_path] [not_used_path] [input_model_path] [output_result_path]
*[not_used_path]:This path is not currently used in the code, so you can input any path.
Load model and test model, support from external input path. Use the format above. 
At present, the external parameter of the second input is not used in the code. You can specify any path.

Other Files:
A CSV file: A table of optimal parameters after grid search
A PNG file: the learning curve
A CSV file saves the optimal parameters after grid search. A PNG picture keeps the learning curve.

3.Evaluation

+ Homogeneity Score
+ Completeness Score
+ V-Measure Score
+ Adjusted Rand Score
+ Adjusted Mutual Score
+ Calinski Harabasz Score
+ Silhouette Score

Now, we use these seven scores to evaluate our model, if you need other scores, you can add it by yourself.
When you run the program, it will output the evaluation results of the models with different parameter combinations.


Homogeneity Score:
A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.


Completeness Score:
A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.


V-Measure Score:
The V-measure is the harmonic mean between homogeneity and completeness.


Adjusted Rand Score:
The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.


Adjusted Mutual Info Score:
The score is an adjustment of the Mutual information(MI) score to account for chance. It accounts for the fact that the MI is generally higher for two clustering with a larger number of clusters, regardless of whether there is actually more information shared. 
This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won't change the score value in any way. 

Calinski Harabasz Score:
The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.


Silhouette Score:
The range of the score is from -1 to 1, the more closer the same kind of the samples are (the more farther the different kinds of the samples are), the higher the score is

Reference to concrete parameters' usage：
https://www.jianshu.com/p/b9528df2f57a
https://blog.csdn.net/u010159842/article/details/78624135

