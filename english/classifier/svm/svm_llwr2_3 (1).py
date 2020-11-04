#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xlrd
import xlwt
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

kernel_list=['linear', 'poly', 'rbf', 'sigmoid']       #kernel function list

def input_float_list():
    """
    Converts each element in the list to a real output

    :return: real number list
    """
    a = input().split()
    b = [float(a[i]) for i in range(len(a))]
    return b

def para_input():                   #input parameters
    print('Please select the classification method of svm：0 means svc，1 means linearsvc!')
    svm_num = int(input())
    print('Select the kernel function type：0 is linear，1 means poly，2 means rbf，3 means sigmoid!')
    kernel_num = int(input())
    print('Please input the value of the penalty factor(C)!')
    C = input_float_list()
    print('Please enter the value of gamma!')
    gamma = input_float_list()
    print('Please enter the degree value!')
    degree = int(input())
    decision_function='ovo'        # dcision_function_shape Changes are not recommended,so comment them out, or uncomment them if you want
    # print('Enter its decision_function_Shape type，default is "ovo"!')
    # decision_function=input()
    print('Please enter a value for Coef0!')
    coef0 = float(input())
    print('Please select the standardized method :0 means no pre-processing, 1 means Robust, 2 means MinMax, 3 means MaxAbs, and 4 means Z-score')
    prepro = int(input())
    return kernel_num, C, gamma, degree, decision_function, coef0, prepro, svm_num

def read_xlsx(ncol_ini, ncol_fin, filename):  # This function does not check the boundary. Please check the boundary value when you use it
    """
    Read the xlsx file data

    :param ncol_ini:
    :param ncol_fin:
    :param filename: file name
    :return: data list
    """
    data_list = []
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    nrows = sheetwork.nrows

    for i in range(nrows):
        new_row_list = []
        for j in range(ncol_ini - 1, ncol_fin):
            data = sheetwork.cell_value(i, j)
            new_row_list.append(data)
        data_list.append(new_row_list)

    return data_list

def xlsx_ncol(filename):
    """
    read xlsx file data label

    :param filename: file name
    :return: data list
    """
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    ncols = sheetwork.ncols

    return ncols

def preprocess(data_list, prepro):
    """
    data pre-processing

    :param data_list: data list
    :param prepro:  process method
    :return: list of processed data
    """
    if prepro == 0:
        data_list_transformed = data_list
    elif prepro == 1:
        data_list_transformed = preprocessing.RobustScaler().fit_transform(data_list)     # Standardized data by IQR.Between the quarter and the third quarter sites
    elif prepro == 2:
        data_list_transformed = preprocessing.MinMaxScaler().fit_transform(data_list)     # Narrow down to [0, 1]
    elif prepro == 3:
        data_list_transformed = preprocessing.MaxAbsScaler().fit_transform(data_list)     # Narrow down to [-1.0, 1.0]
    elif prepro == 4:
        data_list_transformed = preprocessing.StandardScaler().fit_transform(data_list)   # The normalized
    return data_list_transformed

def fit(data_list, label_list, kernel_num, C, gamma, decision_function, degree, coef0, sv_num): # Training data

    if sv_num == 0:
        svc = svm.SVC(kernel=kernel_list[kernel_num], C=C, gamma=gamma, decision_function_shape=decision_function,
                    degree=degree, coef0=coef0, class_weight='balanced', cache_size=500)
        classifier = svc.fit(data_list, label_list)
    elif sv_num == 1:
        linear_svc = svm.LinearSVC(penalty='l2', loss='squared_hinge', tol=gamma,
                                 C=C, class_weight='balanced', max_iter=100000000)
        classifier = linear_svc.fit(data_list, label_list)
    # elif sv_num==2:
    # nu_svc=svm.NuSVR(kernel=kernel_list[kernel_num],C=C,gamma=gamma,degree=degree,coef0=coef0,cache_size=500)
    # classifier=nu_svc.fit(data_list,label_list)
    return classifier

def predict(classifier, data_list):
    """
    predict data

    :param classifier: svm model
    :param data_list: data list
    :return: predict result
    """
    label_list = classifier.predict(data_list)
    return label_list

def evaluate(train_label, test_label, tra_label, tes_label, C, gamma, degree, coef0, prepro, decision_function):     # Evaluation, remove if you want
    print('C:                       ', C)
    print('gamma:                   ', gamma)
    print('degree:                  ', degree)
    print('coef0:                   ', coef0)
    print('preprocess:              ', prepro)
    print('decision_function_shape: ', decision_function)
    print('accuracy:')
    print('train set：', accuracy_score(train_label, tra_label))
    print('test set：', accuracy_score(test_label, tes_label))
    print('precision:')
    print('train set：', precision_score(train_label, tra_label))
    print('test set：', precision_score(test_label, tes_label))
    print('recall:')
    print('train set：', recall_score(train_label, tra_label))
    print('test set：', recall_score(test_label, tes_label))
    print('roc_auc:')
    print('train set：', roc_auc_score(train_label, tra_label))
    print('test set：', roc_auc_score(test_label, tes_label))
    print('f1:')
    print('train set：', f1_score(train_label, tra_label))
    print('test set：', f1_score(test_label, tes_label))

def cross_validation(classifier, data_list, label_list):      # Cross validation method
    scoring = ['accuracy', 'precision', 'recall', 'f1-samples', 'roc_auc']
    scores = cross_val_score(classifier, data_list, label_list, scoring='f1', cv=10)
    print(scores)
    score_mean = scores.mean()
    score_std = scores.std()
    print('score: %0.3f(+/- %0.3f)' % (scores.mean(), scores.std()))
    return score_mean, score_std

def plot_learning_curve(classifier, data_list, label_list):        # Draw a learning curve

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
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.show()

def predict_write_xls(classifier, data_filename, label_filename, prepro):      # Classification, recommended to write as xls file, otherwise possible error
    ncols = xlsx_ncol(data_filename)
    data_list = read_xlsx(1, ncols, data_filename)
    data_list_transformed = preprocess(data_list, prepro)
    label_list = classifier.predict(data_list_transformed)
    label_file = xlwt.Workbook(encoding = 'utf-8')
    label_sheet = label_file.add_sheet(u'sheet1', cell_overwrite_ok = True)

    for i in range(len(data_list)):
        label_sheet.write(i, 0, int(label_list[i]))

    label_file.save(label_filename)


def write_wrong_label(classifier, data_list, label_list, filename):                # Output sample of mis-classification
    predict_label_list = classifier.predict(data_list)
    j = 0
    wrong_file = xlwt.Workbook(encoding = 'utf-8')
    wrong_sheet = wrong_file.add_sheet(u'sheet1', cell_overwrite_ok = True)

    for i in range(len(data_list)):
        if predict_label_list[i] != label_list[i]:
            j = j + 1
            wrong_sheet.write(j, 0, i+1)

    wrong_sheet.write(0, 0, 'Total number of mis-classification')
    wrong_sheet.write(0, 1, j)
    wrong_file.save(filename)

def max_index(list2):
    """
    Look for the maximum's index of two-dimensional list

    :param list2: two-dimensional list
    :return: maximum's index
    """
    a = np.array(list2)
    m, n = a.shape
    index = int(a.argmax())
    x = int(index / n)
    y = index % n
    return x, y

def main():
    print('Please enter the data file path,.xlsx file')
    filename = input()
    print('Please enter process method after training：1/2/3，1 means only training the model and selects the "optimal parameter.”，2 means training model is used to classify the unknown data，3 means classfy the original data and search for error samples by using the trained model')
    pre = int(input())
    if pre == 2:
        print('Please enter the path of the data file to be predicted，.xlsx file')
        predict_data_filename = input()
        print('Please enter the path of the predicted result file，recommended .xls file，otherwise possible error')
        predict_label_filename = input()
    if pre == 3:
        print('Please enter the error sample file output path, recommended xls file')
        wrong_filename = input()
    

    kernel_num, C, gamma, degree, decision_function, coef0, prepro, sv_num = para_input()    # Enter parameters ， C and Gamma can enter multiple values, separated by Spaces

    ncols = xlsx_ncol(filename)                 # Read data and labels

    label_list = []
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    nrows = sheetwork.nrows

    for i in range(nrows):
        label=sheetwork.cell_value(i, 0)
        label_list.append(int(label))

    data_list=read_xlsx(2, ncols, filename)

    data_list_transformed = preprocess(data_list, prepro)          # pre-process

    train_data, test_data, train_label, test_label = train_test_split(data_list_transformed,
                                                                 label_list, test_size=0.1)  # The training set and test set were divided, and the test set proportion was 0.1

    scores_mean = []            #Start the training
    clf_list = []
    for i in range(len(C)):
        new_row_list = []
        new_clf_list = []
        for j in range(len(gamma)):
            # svc=svm.SVC(kernel=kernel_list[kernel_num],decision_function_shape=decision_function,C=C[i],gamma=gamma[j],degree=degree,coef0=coef0,class_weight='balanced')
            # classifier=svc.fit(train_data,train_label)
            classifier = fit(train_data, train_label, kernel_num, C[i],
                             gamma[j], decision_function, degree, coef0, sv_num)
            new_clf_list.append(classifier)
            tra_label = classifier.predict(train_data)
            tes_label = classifier.predict(test_data)
            evaluate(train_label, test_label, tra_label, tes_label, C[i],
                     gamma[j], degree, coef0, prepro, decision_function)
            score_mean, score_std = cross_validation(classifier, data_list_transformed, label_list)
            new_row_list.append(score_mean)
        clf_list.append(new_clf_list)
        scores_mean.append(new_row_list)

    max_C, max_gamma = max_index(scores_mean)                   # Look for the highest score and its subscripts

    classifier = clf_list[max_C][max_gamma]                     # The optimal model/the model with the highest score
    print('The optimal model/the model with the highest score：')
    tra_label = classifier.predict(train_data)
    tes_label = classifier.predict(test_data)
    evaluate(train_label, test_label, tra_label, tes_label, C[max_C],
             gamma[max_gamma], degree, coef0, prepro, decision_function)
    cross_validation(classifier, data_list_transformed, label_list)

    if pre == 2:                                                # classfy new data by using the optimal model 
        predict_write_xls(classifier, predict_data_filename, predict_label_filename, prepro)
    if pre == 3:
        write_wrong_label(classifier, data_list_transformed, label_list, wrong_filename)

    plot_learning_curve(classifier, train_data, train_label)    # Draw a learning curve
   
if __name__ == '__main__':
    main()