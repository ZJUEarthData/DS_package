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

# ***********************************************Adjusting parameter section*************************************************************************
# Model super parameter
CHOICE = 1                                # Data pre-processing approache-- 0.Robust Standardization 1.Normalization 2.Regularization
HIDDEN_LAYER_SIZES = [(100,)]             # The shape of the hidden layer
ACTIVATION = ['relu','identity']          # Activation function{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
SOLVER = ['adam']                         # Weight optimization method {‘lbfgs’, ‘sgd’, ‘adam’}
ALPHA = [0.0001]                          # L2Regularize the severity of punishment
MAX_ITER = [200]                          # Maximum number of iterations
TOL = [1e-4]                              # Error
VERBOSE = [False]                          # Whether to print intermediate results
BATCH_SIZE = ['auto']
RANDOM_STATE = 28
BETA = 0.5                   # BETA<1 focus on precision，BETA>1focus on recall
# file
FEATURE_FILE_PATH = "D:\Pycharm\A_20200629.xlsx" # The location of the feature in the file Everything in the file will be a feature!Please do not include descriptive content such as sample IDs
LABEL_FILE_PATH = "D:\Pycharm\B_20200629.xlsx"   # The location of the labels in the file

# **********************************************************Part of the code***********************************************************************

parameters = {'hidden_layer_sizes':HIDDEN_LAYER_SIZES,
              'activation':ACTIVATION,
             'solver':SOLVER,
             'alpha':ALPHA,
             'batch_size':BATCH_SIZE,
             'max_iter':MAX_ITER,
             'tol':TOL,
             'verbose':VERBOSE}


# Read files, extract features and labels
def get_data(feature_file_path=FEATURE_FILE_PATH,label_file_path=LABEL_FILE_PATH):
    if not os.path.exists(feature_file_path):
        raise ValueError("The feature file does not exist")
    if not os.path.exists(label_file_path):
        raise ValueError("The label file does not exist")
    df_x = pd.read_excel(feature_file_path)
    df_y = pd.read_excel(label_file_path)
    return {'X':df_x,'y':df_y}

# Data preprocessing
def data_process(X,choice):
    if choice==0:
        X=preprocessing.RobustScaler().fit_transform(X)
    elif choice==1:
        X=preprocessing.MinMaxScaler().fit_transform(X)
    elif choice==2:
        X=preprocessing.StandardScaler().fit_transform(X)
    return X

# Partitioning data set
def split_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
    return {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

# Model evaluation
def evaluate_model(clf,X_test,y_test,beta=BETA):
    y_pred = clf.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)        # model precision
    recall = recall_score(y_test, y_pred)             # recall rate
    matrix = confusion_matrix(y_test, y_pred)         #  confusion matrix
    f_score = fbeta_score(y_test, y_pred, beta=beta)  # f_beta
    return {'acc_score':acc_score,'recall':recall,'matrix':matrix,'f_score':f_score,'test_num':len(y_pred)}

# Training neural network classification model
def train_model(X_train,X_test,y_train,y_test):
    # Model training
    mlp = MLPClassifier()
    clf = GridSearchCV(estimator=mlp,param_grid=parameters,cv=5).fit(X_train,y_train)

    #Model evaluation
#     res = evaluate_model(clf,X_test,y_test,beta=BETA)
    return {'results':clf.cv_results_,'clf':clf}

# Save experimental results
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
    # getting data
    print("read the file.........")
    print("The feature file:{}".format(FEATURE_FILE_PATH))
    print("The label file:{}".format(LABEL_FILE_PATH))
    data = get_data(feature_file_path=FEATURE_FILE_PATH,label_file_path=LABEL_FILE_PATH)
    X = data['X'].fillna(0)
    y = data['y'].fillna(0)
    if X.isnull().any().any():
        raise ValueError("Missing data exists in the feature file")
    if y.isnull().any().any():
        raise ValueError("Missing data exists in the label file")
    # data pre-processing
    X = data_process(X,1)
    #Partitioning data set
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=RANDOM_STATE)
    # Get a list of parameters

    # Model training and get results
    print("Model training.........")
    res = train_model(X_train,X_test,y_train,y_test)

    clf = res['clf']                       #the best model
    results = pd.DataFrame(res['results']) #comparison with results
    results.to_csv("{}.csv".format(TIMESTAMP),index=None)
    #Evaluation of optimal model results
    eval_model = evaluate_model(clf,X_test,y_test,beta=BETA)
    # result display
    print("Optimal model evaluation results.........")
    for key,value in eval_model.items():
        print("{}:{}".format(key,value))
    plt.plot(clf.best_estimator_.loss_curve_,c='red', linestyle= '-')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid()
    # plt.show()
    #Sort out the experimental results and write them into a file for saving
    save_res(eval_model,TIMESTAMP)

    return


if __name__ == '__main__':
    main()