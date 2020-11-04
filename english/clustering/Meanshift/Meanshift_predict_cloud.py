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


# **********************************************************Parameter Adjustment Part*************************************************************************
CHOICE = 1   # data pre-processing method -- 0.Robust Standardization  1.normalization 2.Standardization
# file
#FEATURE_FILE_PATH = "/tmp/feature.xlsx"    # file where the features are located 
#LABEL_FILE_PATH = "/tmp/label.xlsx"     # file where the labels are located
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
        raise ValueError("The model file does not exist.")
    Mshift = joblib.load(INPUT_MODEL+'test.pkl')              # call the pre-trained model
    print(Mshift)


    # get data
    print("Read file.........")
    print("Feature file:{}".format(FEATURE_FILE_PATH))

    if not os.path.exists(FEATURE_FILE_PATH):
        raise ValueError("The feature file does not exist.")

    X = pd.read_excel(FEATURE_FILE_PATH).fillna(0)


    for col in X.columns:
        X[col] = X[col].apply(lambda x:str(x).replace(" ",""))
    if X.isnull().any().any():
        raise ValueError("There is missing data in the feature file.")
     # data pre-process
    X_ = data_process(X,CHOICE)

    # data prediction and results visualization 
    y_pred = Mshift.predict(X_)
    y_pred = pd.DataFrame(y_pred,columns=['predict'])

    df = pd.concat([X,y_pred],axis=1)
    df.to_csv(OUTPUT_RESULTS+"{}_results.csv".format(TIMESTAMP),index=None)




if __name__ == '__main__':
    main()
