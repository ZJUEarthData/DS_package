#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import xlrd
#import xlwt
import csv
import codecs
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']       # 核函数表

def input_float_list():
    """
    将列表中的每个元素转化为实数输出

    :return: 实数列表
    """
    a = input().split()
    b = [float(a[i]) for i in range(len(a))]
    return b

def read_txt(filename):
    """
    读取txt文件

    :param filename: 文件名
    :return: 文件内容的列表
    """
    pos=[]
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            new_row_list = [float(i) for i in lines.split()]
            pos.append(new_row_list)
            pass
        pass
    return pos

def para_file_input(pos):
    """
    读取文件中的必要的参数

    :param pos: 文件内容列表
    :return: 返回所需参数
    """
    svm_num = int(pos[0][0])
    kernel_num = int(pos[1][0])
    C = np.linspace(pos[2][0], pos[2][1], int(pos[2][2]))
    gamma = np.linspace(pos[3][0], pos[3][1], int(pos[3][2]))
    degree = pos[4][0]
    coef0 = pos[5][0]
    pre = pos[6][0]
    prepro = pos[7][0]
    return svm_num, kernel_num, C, gamma, degree, coef0, pre, prepro

def read_xlsx(ncol_ini, ncol_fin, filename):      # 本函数没有设置边界检查，还请输入时注意检查边界值
    """
    读取xlsx文件数据

    :param ncol_ini:
    :param ncol_fin:
    :param filename: 文件名
    :return: 数据列表
    """
    data_list = []
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    nrows = sheetwork.nrows

    for i in range(nrows):
        new_row_list=[]
        for j in range(ncol_ini-1, ncol_fin):
            data = sheetwork.cell_value(i, j)
            new_row_list.append(data)
        data_list.append(new_row_list)
    
    return data_list

def xlsx_ncol(filename):
    """
    读取xlsx文件数据标签

    :param filename: 文件名
    :return: 数据列表
    """
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    ncols = sheetwork.ncols

    return ncols

def preprocess(data_list, prepro):
    """
    数据预处理

    :param data_list: 数据列表
    :param prepro:  处理方式
    :return: 处理后的数据列表
    """
    if prepro == 0:
        data_list_transformed = data_list
    elif prepro == 1:
        data_list_transformed = preprocessing.RobustScaler().fit_transform(data_list)     # 通过IQR标准化数据，即四分之一和四分之三分位点之间
    elif prepro == 2:
        data_list_transformed = preprocessing.MinMaxScaler().fit_transform(data_list)     # 缩放到区间 [0, 1]
    elif prepro == 3:
        data_list_transformed = preprocessing.MaxAbsScaler().fit_transform(data_list)     # 缩放到区间[-1.0, 1.0]内
    elif prepro == 4:
        data_list_transformed = preprocessing.StandardScaler().fit_transform(data_list)   # 归一化
    return data_list_transformed

def fit(data_list, label_list, kernel_num, C, gamma, decision_function, degree, coef0, sv_num): # 训练数据

    if sv_num == 0:
        svc = svm.SVC(kernel=kernel_list[kernel_num], C=C, gamma=gamma, decision_function_shape=decision_function,
                    degree=degree, coef0=coef0, class_weight='balanced', cache_size=500)
        classifier = svc.fit(data_list, label_list)
    elif sv_num == 1:
        linear_svc = svm.LinearSVC(penalty='l2', loss='squared_hinge', tol=gamma,
                                 C=C, class_weight='balanced', max_iter=100000000)
        classifier = linear_svc.fit(data_list, label_list)
    return classifier

def predict(classifier, data_list):
    """
    对数据进行预测

    :param classifier: svm模型
    :param data_list: 数据集
    :return: 预测结果
    """
    label_list = classifier.predict(data_list)
    return label_list

def evaluate(train_label, test_label, tra_label, tes_label, C, gamma, degree, coef0, prepro, decision_function):     # 评价，如果嫌麻烦可以自行去掉
    print('C:                       ', C)
    print('gamma:                   ', gamma)
    print('degree:                  ', degree)
    print('coef0:                   ', coef0)
    print('preprocess:              ', prepro)
    print('decision_function_shape: ', decision_function)
    print('accuracy:')
    print('训练集：', accuracy_score(train_label, tra_label))
    print('测试集：', accuracy_score(test_label, tes_label))
    print('precision:')
    print('训练集：', precision_score(train_label, tra_label))
    print('测试集：', precision_score(test_label, tes_label))
    print('recall:')
    print('训练集：', recall_score(train_label, tra_label))
    print('测试集：', recall_score(test_label, tes_label))
    print('roc_auc:')
    print('训练集：', roc_auc_score(train_label, tra_label))
    print('测试集：', roc_auc_score(test_label, tes_label))
    print('f1:')
    print('训练集：', f1_score(train_label, tra_label))
    print('测试集：', f1_score(test_label, tes_label))

def cross_validation(classifier, data_list, label_list):      # 交叉验证方法
    scoring = ['accuracy', 'precision', 'recall', 'f1-samples', 'roc_auc']
    scores = cross_val_score(classifier, data_list, label_list, scoring='f1', cv=10)
    print(scores)
    score_mean = scores.mean()
    score_std = scores.std()
    print('score: %0.3f(+/- %0.3f)' % (scores.mean(), scores.std()))
    return score_mean, score_std


def plot_learning_curve(classifier, data_list, label_list, output_png):        # 绘制学习曲线

    train_sizes, train_scores, test_scores = learning_curve(classifier, data_list, label_list, cv=10, n_jobs=-1,
                                                        train_sizes=np.linspace(0.1,1.0,50), scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='training')
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1, color='r')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='b', label='cross_validation')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, alpha=0.1, color='b')
    plt.xlabel('train sizes')
    plt.ylabel('f1')
    plt.legend(loc='best')
    # plt.show()
    fig = plt.savefig(output_png)
    return fig


def predict_write_xls(classifier, predict_data_list, output_xls, prepro):      # 分类，建议写出为.xls文件，否则可能出错
    data_list_transformed = preprocess(predict_data_list, prepro)
    label_list = classifier.predict(data_list_transformed)

    wrong_list = []
    f = codecs.open(output_xls, 'w', 'utf-8')
    writer = csv.writer(f)
    for i in range(len(label_list)):
        # writer.writerow(str(label_list[i]))
        new_row_list = []
        new_row_list.append(label_list[i])
        wrong_list.append(new_row_list)

    writer.writerows(wrong_list)

    f.close()

def write_wrong_label(classifier, data_list, label_list, output_xls):      # 输出错误分类样本
    predict_label_list=classifier.predict(data_list)

    wrong_list = []
    f = codecs.open(output_xls, 'w', 'utf-8')
    writer = csv.writer(f)
    for i in range(len(data_list)):
        if predict_label_list[i] != label_list[i]:
            new_row_list = []
            new_row_list.append(i+1)
            wrong_list.append(new_row_list)

    writer.writerows(wrong_list)
    f.close()

def max_index(list2):
    """
    寻找二维list最大值下标

    :param list2: 二维列表
    :return: 最大值下标
    """
    a = np.array(list2)
    m, n = a.shape
    index = int(a.argmax())
    x = int(index / n)
    y = index % n
    return x, y

def main(argv):

    decision_function = 'ovo'             # 以下为输入一些参数，其中C和gamma可以输入多个值，用空格分开
    filename = argv[2]
    options = read_txt(argv[1])
    output_png = argv[4]
    output_xls = argv[3]
    svm_num, kernel_num, C, gamma, degree, coef0, pre, prepro = para_file_input(options)

    ncols = xlsx_ncol(filename)           # 读入数据和标签

    label_list = []
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    nrows = sheetwork.nrows

    for i in range(nrows):
        label = sheetwork.cell_value(i, 0)
        label_list.append(int(label))

    data_list=read_xlsx(2, ncols, filename)
    
    predict_data_list = []
    train_data_list = []
    train_label_list = []
    if pre == 2:
        for i in range(nrows):
            if label_list[i] == 0:
                predict_data_list.append(data_list[i])
            else:
                train_data_list.append(data_list[i])
                train_label_list.append(label_list[i])

        data_list = train_data_list
        label_list = train_label_list


    data_list_transformed = preprocess(data_list, prepro)

    train_data, test_data, train_label, test_label = train_test_split(data_list_transformed, label_list, test_size=0.1)

    scores_mean = []                          # 以下开始训练
    clf_list = []
    for i in range(len(C)):
        new_row_list = []
        new_clf_list = []
        for j in range(len(gamma)):
            classifier = fit(train_data, train_label, kernel_num, C[i], gamma[j],
                           decision_function, degree, coef0, svm_num)
            new_clf_list.append(classifier)
            tra_label = classifier.predict(train_data)
            tes_label = classifier.predict(test_data)
            evaluate(train_label, test_label, tra_label, tes_label, C[i],
                     gamma[j], degree, coef0, prepro, decision_function)
            score_mean, score_std=cross_validation(classifier, data_list_transformed, label_list)
            new_row_list.append(score_mean)
        clf_list.append(new_clf_list)
        scores_mean.append(new_row_list)

    max_C, max_gamma = max_index(scores_mean)  # 寻找最高评分及其下标

    classifier = clf_list[max_C][max_gamma]   # 最优模型/分数最高的模型
    print('最优模型/分数最高的模型：')
    tra_label = classifier.predict(train_data)
    tes_label = classifier.predict(test_data)
    evaluate(train_label, test_label, tra_label, tes_label, C[max_C],
             gamma[max_gamma], degree, coef0, prepro, decision_function)
    cross_validation(classifier, data_list_transformed, label_list)

    if pre == 2:                              # 应用最优模型完成新数据的分类
        predict_write_xls(classifier, predict_data_list, output_xls, prepro)
    if pre == 3:
        write_wrong_label(classifier, data_list_transformed, label_list, output_xls)

    plot_learning_curve(classifier, train_data, train_label, output_png)       # 绘制学习曲线
   
if __name__ == '__main__':
    main(sys.argv)
