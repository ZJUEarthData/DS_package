LOF+ Module
autho:Arif Khan
time:Oct 12，2021

This is a python module to subset geochemical dataset into Inliers and Outliers by Local Outlier Factor analysis based on feature bagging. Subsequent plotting of result is also available with 'plot' module.

Dependencies: NumPy, Pandas, Plotly Express, Matplotlib, SKLearn, Math

	random_subspace(features,subspaces, min_features, max_features)
	A function to generate a list of subsets of features in the dataset based on user preference. Unless max and min features are supplied, feature subsets length between maximum possible features and half of the maximum features is generated.
	
	Parameters:
	features (list): A list of features.
	subspaces (int): An integer denoting the number of non-unique subspaces to be created.
	min_features (int): Minimum number of features to be used to create subspaces. (default= None. half the total number of features.)
	max_features (int): Maximum number of features to be used to create subspaces. (default= None. the total number of features.)
	
	Returns:
	list: A nested list of features.

detect(data, no_of_subspaces,tot_feature,n_neighbors, contamination, min_features, max_features, separate_df)
	To perform Local Outlier Factor outlier analysis. Modies the original data frame inplace by adding a 'label' column having the analysis result. Default values are detailed below.
	
	Parameters:
	data (DataFrame): A DataFrame.
	no_of_subspaces (int): An integer denoting the number of non-unique subspaces to be created.
	tot_feature (list): A list of all features to be used for analysis.
	n_neighbors (int): Number of neighbours. (default= 50)
	contamination (float): Number between 0 and 0.5 denoting proportion of outliers. (default= 'auto')
	min_features (int): Minimum number of features to be used to create subspaces. (default= None. uses half the total number of features.)
	max_features (int): Maximum number of features to be used to create subspaces. (default= None. uses the total number of features.)
	separate_df(boolean): Whether to create separate dataframes for outliers and inliers.
	
	Returns:
	None: if separate_df=False.
	tuple: A tuple of two dataframes outliers and inliers respectively in the format (outliers,inliers) if separate_df=True.

plot(data,features,title,col_name,save_fig)
	Generates a plot of the analysis based on label. Uses Plotly express to generate interactive plots. Plots can be 2 dimensional or 3 dimensional based on number of features and exported as HTML file.
	
	Parameters:
	data (DataFrame): A DataFrame.
	features (list): A list of length 2 or 3 or less of corresponding plot axes.
	title (string): Plot name. (default: 'LOF Plot')
	col_name (string): Name of column in the dataframe having the outlier labels. (default: 'label')
	save_fig (boolean): whether to save the figure as an external HTML file. (default: True)
	
	Returns:
	fig: A 3D plotly scatter plot
	html: 3D plotly scatter plot file