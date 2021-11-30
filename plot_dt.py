from sklearn import tree
from matplotlib import pyplot as plt
from dt import process_dt
from pandas import read_csv

x_train = read_csv('x_train.csv')
features = x_train.columns
x_train = x_train.to_numpy()
x_test = read_csv('x_test.csv').to_numpy()
y_train = read_csv('y_train.csv').to_numpy()
y_test = read_csv('y_test.csv').to_numpy()


def main():
    clf = process_dt(x_train, x_test, y_train, y_test)
    with open("dt.txt", "w") as f:
        f.write(tree.export_text(clf, feature_names=(list(features))))


main()
