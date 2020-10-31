#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report
def stratified(df):
    """
    基于kmeans聚类结果的分层抽样

    :param df:聚类结果
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["km_clustering_label"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]
    df = pd.DataFrame(strat_train_set)
    df.to_csv('test4_result.csv')
    df2 = pd.DataFrame(strat_test_set)
    df2.to_csv('test4_result.csv', mode='a', header=False)

def fit(x_data, y_label):       # 训练
    x_train_val, x_test, y_train_val, y_test = train_test_split(x_data, y_label, test_size=0.3, random_state=0)
    # y_train_val.shape, y_test.shape

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=1)
    # y_train.shape, y_val.shape

    svc_val = SVC(kernel='rbf', C=250.075, gamma=0.1)
    svc_val.fit(x_train, y_train)
    score_val = svc_val.score(x_val, y_val)
    print('SVM Score: %.3f' % score_val)
    return svc_val, x_train, y_train, x_train_val, y_train_val, x_test, y_test

def l_curve(train_sizes, train_scores, test_scores):       # 绘制学习曲线
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    # plt.title('AUC')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()
    # plt.savefig(output_png)

def main()
    df = pd.read_csv("test_result_newxk2000_4簇.csv")    # 查看使用kmeans聚类后的分类标签值，两类
    df['km_clustering_label'].hist()

    stratified(df)

    orig_data = pd.read_csv('test4_result.csv')         # 读取抽样结果
    # orig_data = pd.read_excel('test_newxk2000.xlsx')
    orig_data.dropna(inplace=True)                      # 删除包含缺失值的行
    # orig_data
    x_orig_data = orig_data.drop('TRUE VALUE', axis=1)
    y_orig_label = orig_data['TRUE VALUE']
    y_label = y_orig_label.replace(-1, 0)

    process = preprocessing.StandardScaler()  # 归一化
    x_data = process.fit_transform(x_orig_data)

    svc_val, x_train, y_train, x_train_val, y_train_val, x_test, y_test = fit(x_data, y_label)

    train_sizes, train_scores, test_scores = learning_curve(svc_val, x_train, y_train, cv=10,
                                                            n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    l_curve(train_sizes, train_scores, test_scores)

    svc_test = SVC(kernel='rbf', C=250.075, gamma=0.1)
    svc_test.fit(x_train_val, y_train_val)
    y_test_pred = svc_test.predict(x_test)

    print('Accuracy: %.3f' % accuracy_score(y_test, y_test_pred))       # 输出预测准确率
    print('ROC AUC: %.3f' % roc_auc_score(y_test, y_test_pred))

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_test_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_test_pred))
    print('F1 Score: %.3f' % f1_score(y_true=y_test, y_pred=y_test_pred))

    print(classification_report(y_true=y_test, y_pred=y_test_pred))

    confmat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    # print(confmat)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

if __name__ == '__main__':
    main()

