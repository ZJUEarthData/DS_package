from sklearn.model_selection import train_test_split
import sys
import os
import joblib
import warnings
from mlp_training_cloud import get_data,data_process
import pandas as pd
import matplotlib.pyplot as plt
import datetime
warnings.filterwarnings("ignore")


# **********************************************************调参部分*************************************************************************
CHOICE = 1   # 数据预处理方式-- 0.Robust标准化  1.归一化 2.标准化
# 文件
#FEATURE_FILE_PATH = "/tmp/feature.xlsx" # 特征所在文件 文件中所有
#LABEL_FILE_PATH = "/tmp/label.xlsx"   # 标签所在文件

def main():
    FEATURE_FILE_PATH = sys.argv[1]
    LABEL_FILE_PATH = sys.argv[2]
    INPUT_MODEL = sys.argv[3]
    OUTPUT_RESULTS = sys.argv[4]

    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'","")
#    base = os.getcwd()
    base = INPUT_MODEL
    dir = os.path.join(base,"test.pkl")
    if not os.path.exists(dir):
        raise ValueError("模型文件不存在")
    Mshift = joblib.load(INPUT_MODEL+'test.pkl')             # 调用预训练好的模型
    print(Mshift)


    # 获取数据
    print("读取文件.........")
    print("特征文件:{}".format(FEATURE_FILE_PATH))

    if not os.path.exists(FEATURE_FILE_PATH):
        raise ValueError("特征文件不存在")

    X = pd.read_excel(FEATURE_FILE_PATH).fillna(0)


    for col in X.columns:
        X[col] = X[col].apply(lambda x:str(x).replace(" ",""))
    if X.isnull().any().any():
        raise ValueError("特征文件存在缺失数据")
    # 数据预处理
    X_ = data_process(X,CHOICE)

    # 模型预测并可视化结果
    y_pred = Mshift.predict(X_)
    y_pred = pd.DataFrame(y_pred,columns=['predict'])

    df = pd.concat([X,y_pred],axis=1)
    df.to_csv(OUTPUT_RESULTS+"{}_results.csv".format(TIMESTAMP),index=None)




if __name__ == '__main__':
    main()