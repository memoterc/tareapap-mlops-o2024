import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score




def main() -> None:
    
    X_test = pd.read_csv(r"./data/credit_pred.csv")
    train_data = pd.read_csv(r"./data/credit_train.csv")

    x_train = train_data.drop(columns=["Y"])
    y_train = train_data["Y"]

    # Using DMatrix cause why tf no
    dm_train_data = xgb.DMatrix(x_train, label = y_train)
    X_test = xgb.DMatrix(X_test)

    

    
    pass








if __name__ == "__main__":
    main()