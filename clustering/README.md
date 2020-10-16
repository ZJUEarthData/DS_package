聚类使用说明
Instructions for Clustering Algorithm

一、简介（Brief Introduction）
包中所有聚类算法都按照网格搜索最优参数，训练模型，保存结果的顺序运行。包中的XX.py, XX_class.py, XX_cloud.py文件是三种不同运行方式的源代码，可以根据需要进行选择。
These codes use GridSearchCV to find optimal parameters and train the model.
There are four algorithms, and every algorithm has three files(XX.py, XX_class.py, XX_cloud.py) that can run in different ways, and each of these files requires a data set and a hyper-parameter table as input.
(XX refers to one of AP, kmeans, Meanshift, svm)


二、文件说明（Filespec）
Data set（数据集）: 
An excel sheet, the file “test4.xlsx” is a sample.
一个excel表，储存训练数据。


Hyper-parameter table（超参数表）:
An excel sheet, it contains the value of the hyper-parameter you want to test. The leftmost column is the parameter name, on the right is the value of the parameter you want to test. The file “para.xlsx” is a sample.
一个excel表，每行最左侧是超参数名称，右侧是需要进行网格搜索的数值。


XX.py: 
This file can be run directly through the terminal, but make sure the name of the data set and the hyper-parameter table match their names in the code.
可以直接运行的源代码，但是要注意数据集的名称、超参数表的名称要与代码中的名称一致。


XX_class.py:
This file puts all the functions used by XX.py in one class. You can import this class when you want to use it.
所有XX.py需要用到的函数都被封装到这个类中。需要使用时，可以在另一个python文件中import这个类。



XX_cloud.py:
This file can be run directly through the terminal using the following format:
python XX_cloud.py [parameter_file_path] [data_file_path] [output_model_path] [output_result_path]
该文件支持外部输入路径，在终端中运行格式如上，要依次输入超参数表路径、数据集路径、模型输出路径、结果输出路径。下面是使用AP算法的样例。
Sample:
python AP_cloud.py ./para.xlsx ./test4.xlsx ./ ./
( ‘./’ means current folder)


test.pkl:
The best model after training will be saved in this file. XX_predict_cloud.py will load this model to predict the test set.
训练后的模型将会保存在这个文件中，XX_predict_cloud.py会载入这个模型，对测试集进行预测


XX_predict_cloud.py:
Loading model and making prediction. You should use the following format:
python AP_predict_cloud.py [data_file_path] [not_used_path] [input_model_path] [output_result_path]
*[not_used_path]:This path is not currently used in the code, so you can input any path.
载入模型和测试模型，支持从外部输入路径。使用格式如上。目前第二个输入的外部参数在代码中未使用，可以任意指定一个路径。


Other Files(其他文件):
A CSV file: A table of optimal parameters after grid search
A PNG file: the learning curve
一个csv文件保存了网格搜索后的最优参数。
一个png图片保存了学习曲线。


三、评价标准（Evaluation）

+ Homogeneity Score
+ Completeness Score
+ V-Measure Score
+ Adjusted Rand Score
+ Adjusted Mutual Score
+ Calinski Harabasz Score
+ Silhouette Score

Now, we use these scores to evaluate our model, if you need other scores, you can add it by yourself.
When you run the program, it will output the evaluation results of the models with different parameter combinations.
目前使用如上7个评价标准，如果需要其他的可以自己添加。
每次在终端运行时，会把所有参数搭配的模型的评价结果都输出出来。


Homogeneity Score:(每个群集是否只包含单个类的成员)
A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.


Completeness Score:(给定类的所有成员都分配给同一个群集)
A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster.


V-Measure Score:(上两个标准的调和平均)
The V-measure is the harmonic mean between homogeneity and completeness.


Adjusted Rand Score:(衡量的是两个数据分布的吻合程度)
The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.


Adjusted Mutual Info Score:(利用基于互信息的方法来衡量聚类效果需要实际类别信息，值越大意味着聚类结果与真实情况越吻合。)
The score is an adjustment of the Mutual information(MI) score to account for chance. It accounts for the fact that the MI is generally higher for two clustering with a larger number of clusters, regardless of whether there is actually more information shared. 
This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won't change the score value in any way. 

Calinski Harabasz Score:(方差比指数)
The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.


Silhouette Score:(轮廓系数的取值范围是[-1,1]，同类别样本距离越相近，不同类别样本距离越远，分数越高)
The range of the score is from -1 to 1, the more closer the same kind of the samples are (the more farther the different kinds of the samples are), the higher the score is

Reference to concrete parameters' usage(具体参数说明还可以参考下面的资料)：
https://www.jianshu.com/p/b9528df2f57a
https://blog.csdn.net/u010159842/article/details/78624135

