import pandas as pd
from globals import *
from sklearn.tree import DecisionTreeClassifier


def main():
    # Read dataset
    x_train = pd.read_csv('xTrain.csv')
    y_train = pd.read_csv('yTrain.csv')
    x_test = pd.read_csv('xTest.csv')
    y_test = pd.read_csv('yTest.csv')

    # Initialize decision tree classifier
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)


if __name__ == "__main__":
    main()
