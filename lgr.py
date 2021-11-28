import pandas as pd
from globals import *
from sklearn.linear_model import LogisticRegression


def main():
    # Read dataset
    x_train = pd.read_csv('xTrain.csv')
    y_train = pd.read_csv('yTrain.csv')
    x_test = pd.read_csv('xTest.csv')
    y_test = pd.read_csv('yTest.csv')

    # Initialize logistic regression model
    lgr = LogisticRegression(random_state=RANDOM_STATE).fit(x_train, y_train)


if __name__ == "__main__":
    main()
