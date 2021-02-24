from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# **********************************************************调参部分*************************************************************************
# 模型超参数
CHOICE = 1                                # 数据预处理方式-- 0.Robust标准化  1.归一化 2.正则化
HIDDEN_LAYER_SIZES = [(100,)]             # 隐层的形状
ACTIVATION = ['relu','identity']          # 激活函数{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
SOLVER = ['adam']                         # 权重优化方法 {‘lbfgs’, ‘sgd’, ‘adam’}
ALPHA = [0.0001]                          # L2正则化惩罚力度
MAX_ITER = [200]                          # 最大迭代次数
TOL = [1e-4]                              # 误差
VERBOSE = [False]                          # 是否打印中间结果
BATCH_SIZE = ['auto']
RANDOM_STATE = 28
BETA = 0.5                   # BETA<1侧重查准率，BETA>1侧重查全率
# 文件
FEATURE_FILE_PATH = "D:\Pycharm\A_20200629.xlsx" # 特征所在文件 文件中所有内容都会当作特征！请不要包含样本ID等描述性内容
LABEL_FILE_PATH = "D:\Pycharm\B_20200629.xlsx"   # 标签所在文件

# **********************************************************代码部分*************************************************************************

parameters = {'hidden_layer_sizes':HIDDEN_LAYER_SIZES,
              'activation':ACTIVATION,
             'solver':SOLVER,
             'alpha':ALPHA,
             'batch_size':BATCH_SIZE,
             'max_iter':MAX_ITER,
             'tol':TOL,
             'verbose':VERBOSE}


# 读取文件，提取特征和标签
def get_data(feature_file_path=FEATURE_FILE_PATH,label_file_path=LABEL_FILE_PATH):
    if not os.path.exists(feature_file_path):
        raise ValueError("特征文件不存在")
    if not os.path.exists(label_file_path):
        raise ValueError("标签文件不存在")
    df_x = pd.read_excel(feature_file_path)
    df_y = pd.read_excel(label_file_path)
    return {'X':df_x,'y':df_y}

# 数据预处理
def data_process(X,choice):
    if choice==0:
        X=preprocessing.RobustScaler().fit_transform(X)
    elif choice==1:
        X=preprocessing.MinMaxScaler().fit_transform(X)
    elif choice==2:
        X=preprocessing.StandardScaler().fit_transform(X)
    return X

# 划分数据集
def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
    return {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

# 模型评估
def evaluate_model(clf,X_test,y_test,beta=BETA):
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)        # 模型精度
    recall = recall_score(y_test, y_pred)             # 召回率
    matrix = confusion_matrix(y_test, y_pred)         # 混淆矩阵
    f_score = fbeta_score(y_test, y_pred, beta=beta)  # f_beta
    return {'acc_score':acc_score,'recall':recall,'matrix':matrix,'f_score':f_score,'test_num':len(y_pred)}

# 训练神经网络分类模型
def train_model(X_train,X_test,y_train,y_test):
    # 模型训练
    mlp = MLPClassifier()
    clf = GridSearchCV(estimator=mlp,param_grid=parameters,cv=5).fit(X_train,y_train)

    #模型评估
#     res = evaluate_model(clf,X_test,y_test,beta=BETA)
    return {'results':clf.cv_results_,'clf':clf}

# 保存实验结果
def save_res(eval_model,TIMESTAMP):
    res = {}
    res['TIMESTAMP'] = TIMESTAMP
    res['acc_score'] = [eval_model['acc_score']]
    res['recall'] = [eval_model['recall']]
    res['f_beta'] = [BETA]
    res['f_score'] = [eval_model['f_score']]
    (tn, fp, fn, tp) = eval_model['matrix'].ravel()
    res['tn'] = [tn]
    res['fp'] = [fp]
    res['fn'] = [fn]
    res['tp'] = [tp]
    res['test_num'] = [eval_model['test_num']]


    df = pd.DataFrame(res)
    if os.path.exists("result.csv"):
        df.to_csv("result.csv",mode='a',index=None,header=None)
    else:
        df.to_csv("result.csv",index=None)



def main():
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'","")
    # 获取数据
    print("读取文件.........")
    print("特征文件:{}".format(FEATURE_FILE_PATH))
    print("标签文件:{}".format(LABEL_FILE_PATH))
    data = get_data(feature_file_path=FEATURE_FILE_PATH,label_file_path=LABEL_FILE_PATH)
    X = data['X'].fillna(0)
    y = data['y'].fillna(0)
    if X.isnull().any().any():
        raise ValueError("特征文件存在缺失数据")
    if y.isnull().any().any():
        raise ValueError("标签文件存在缺失数据")
    # 数据预处理
    X = data_process(X,1)
    # 划分数据集
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=RANDOM_STATE)
    # 获取参数列表

    # 模型训练并且拿到结果
    print("模型训练.........")
    res = train_model(X_train,X_test,y_train,y_test)

    clf = res['clf']                       #最佳模型
    results = pd.DataFrame(res['results']) #结果对比
    results.to_csv("{}.csv".format(TIMESTAMP),index=None)
    # 最佳模型结果评估
    eval_model = evaluate_model(clf,X_test,y_test,beta=BETA)
    # 结果展示
    print("最佳模型评估结果.........")
    for key,value in eval_model.items():
        print("{}:{}".format(key,value))
    plt.plot(clf.best_estimator_.loss_curve_,c='red', linestyle= '-')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid()
    # plt.show()
    # 整理实验结果写入文件保存
    save_res(eval_model,TIMESTAMP)

    return


if __name__ == '__main__':
    main()