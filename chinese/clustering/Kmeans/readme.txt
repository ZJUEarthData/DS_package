该kmeans文件夹共四个.py文件和一个.ipynb，分别为：
kmeans.py
kmeans_class.py
kmeans_cloud.py
kmeans_predict_cloud.py
test_class.ipynb


kmeans.py
该文件主要思想是采用kmeans聚类方法对待分类数据实现非监督学习式的分类，即初始化k个聚类中心点并逐步计算每个数据点到每个中心点的距离，并选取最近的中心点所在类作为一类，运行完所有数据点之后用每一类的平均值作为新的中心点，重新进行上述归类运算。其中，
KmeansGridsearch函数用来将模型参数输入到该算法模型里，并返回到模型集合；
get_marks函数用来提供数据集的各种表现分数；
plotit函数用来视图化每个不同参数的模型的表现分数；
read_para函数用来读取外部输入的模型参数；
plot_learning_curve函数用来绘制学习曲线；
main()函数用来整体实现并保存模型参数。

待分析数据文件选自test4.xlsx。
评价标准采用adjusted_rand_score分数来选取最佳表现模型参数。
模型可调参数通过read_para()函数来实现外部para.xlsx文件读取。

kmeans_class.py
该文件内部函数同kmeans.py一样，不同点是建立类（class）GeoKmeans来实现该类所包含的所有函数的外部调用。

kmeans_cloud.py
该文件内容与kmeans.py相同，用来在外部实现整个文件的运行。

kmeans_predict_cloud.py
该文件用来读取保存好的最佳模型，并预测新的数据分类结果。

test_class.ipynb
该文件可用来实现notebook上的运行测试。