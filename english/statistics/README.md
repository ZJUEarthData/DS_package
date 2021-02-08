# Statistics


## Module Name
1. earth_data_plot
2. monte_carlo_simulator


## Application Scenario 

When you have a data set with missing values and you applied a missing value imputation method (i.e. Median Imputation, Mean Imputation, KNN Imputation, etc) to you data. You have your data filled but you definitely don't want to change your original data distribution, then how do you know if it's changed or not? 

These two modules provides two statistical methods that use either probability plots or hypothesis tests to help you decide if the imputed data is significantly different from the original data. If the data distribution is changed after imputation, you may need to use another imputation method instead.

  
## Brief Introduction on Methods in the Two Modules

`monte_carlo_simulator.monte_carlo_simulator` -  This method utilize hypothesis test and monte carlo simulation to test if the imputed data distributions are significantly different from the original data distribtutions. The output of this method will give you a list of column names of the data that reject the null hypothesis, in other words, the distribution of those columns are significantly different from the original distribution, and you may need to use another method to impute your data.

`earth_data_plot.probability_plot` - This method use probability plots to display the similarity between the two distributions (before / after imputaion). If the probability plot of column A shows an approximate diagonal line, then it means the imputed distribution of column A does not significantly deviate from the original distribution.

`earth_data_plot.correlation_plot`, `earth_data_plot.distribution_plot`, `earth_data_plot.logged_distribution_plot` are all plotting methods that help you explore the distribution of the data.

Detailed annotation can be found under each method.



## How to use these two modules?

Please refer to this [demo](https://github.com/hudan42/Statistics/blob/Dan/demo_test_imputation_hypo%26plots.ipynb)


## Tutorial Video

Below are the detailed tutorial video about these two modules.
 
- [Tutorial No.1](https://www.bilibili.com/video/BV1rt4y1k7rc)
- [Tutorial No.2](https://www.bilibili.com/video/BV1gX4y1u7XM)
- [Tutorial No.3](https://www.bilibili.com/video/BV1vK4y1V7Wp)












