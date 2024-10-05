from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import pickle


def main():

    train_data = pd.read_csv(r"./data/credit_train.csv")
    X = train_data.drop("Y", axis=1)
    Y = train_data["Y"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size = 0.3,
        random_state = 69
    )

    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth = 10,
        random_state = 42
    )

    model.fit(x_train, y_train)

    y_hat_train = model.predict(x_train)
    y_hat_test = model.predict(x_test)

    f1_train = f1_score(y_train, y_hat_train)
    f1_test = f1_score(y_test, y_hat_test)

    acc_train = accuracy_score(y_train, y_hat_train)
    acc_test = accuracy_score(y_test, y_hat_test)

    cvscore = cross_val_score(model, X, Y, cv = 3, scoring = "f1")

    with open("./models/rfmodel.pkl", "wb") as file:
        pickle.dump(model, file)


"""
    print(f"f1 score for train: {f1_train}")
    print(f"f1 score for test: {f1_test}")

    print(f"acc score for train: {acc_train}")
    print(f"acc score for test: {acc_test}")

    print(f"Cross validation scores: {cvscore}")
    print(f"Mean f1 : {cvscore.mean()}")
"""


if __name__ == "__main__":
    main()