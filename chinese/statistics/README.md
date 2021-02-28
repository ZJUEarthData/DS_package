# Statistics


## 模块名
1. earth_data_plot 

2. monte_carlo_simulator


##应用场景 

当你有一个缺失值的数据集，并对你的数据应用缺失值估算方法(即中位数估算、平均值估算、KNN估算等)。你已经填好了数据，但又不想改变原始数据分布，那你怎么知道它是否改变了呢?
这两个模块提供了两种统计方法，使用概率图或假设检验来帮助您确定输入数据是否与原始数据显著不同。如果估算后数据分布发生了变化，则可能需要使用另一种估算方法。
  
## 两个模块的方法简介

`monte_carlo_simulator.monte_carlo_simulator` -  该方法利用假设检验和蒙特卡罗模拟来检验输入的数据分布是否与原始数据分布有显著差异。该方法的输出将为您提供拒绝零假设的数据的列名列表，换句话说，这些列的分布与原始分布有很大的不同，您可能需要使用另一种方法来输入数据。
`earth_data_plot.probability_plot` - 该方法使用概率图来显示两个分布(归算前/归算后)之间的相似性。如果A列的概率图显示出一条近似的对角线，则说明输入的A列的分布与原分布没有明显的偏离。
`earth_data_plot.correlation_plot`, `earth_data_plot.distribution_plot`, `earth_data_plot.logged_distribution_plot`都是帮助您探索数据分布的绘图方法。

在每个方法下都可以找到详细的注释。

 ##如何使用这两个模块?

请参考以下[演示](https://github.com/hudan42/Statistics/blob/Dan/demo_test_imputation_hypo%26plots.ipynb)


## 教程视频

下面是关于这两个模块的详细教程视频.
 
- [Tutorial No.1](https://www.bilibili.com/video/BV1rt4y1k7rc)
- [Tutorial No.2](https://www.bilibili.com/video/BV1gX4y1u7XM)
- [Tutorial No.3](https://www.bilibili.com/video/BV1vK4y1V7Wp)






















