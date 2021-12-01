import pandas as pd
from numpy import ravel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


def main():
    # Read dataset
    x_train = pd.read_csv('../x_train.csv').to_numpy()
    y_train = pd.read_csv('../y_train.csv').to_numpy()
    x_test = pd.read_csv('../x_test.csv').to_numpy()
    y_test = pd.read_csv('../y_test.csv').to_numpy()

    # Initialize decision tree classifier
    dt = DecisionTreeClassifier()
    # grid:
    # number of trees
    splitter = ["best", "random"]
    # number of depth
    max_depth = list(range(10, 21))
    # msl
    min_samples_leaf = list(range(6, 17, 2))
    # weight fraction leaf
    # min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]
    # # max_features
    # max_features = ["auto", "log2", "sqrt"]
    # # max_leaf_nodes
    # max_leaf_nodes = [10, 20, 30, 40, 50]

    # create grid
    random_grid = {'splitter': splitter,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_samples_leaf}

    dt_random = RandomizedSearchCV(estimator=dt, param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2, random_state=42,
                                   n_jobs=-1)

    dt_random.fit(x_train, ravel(y_train))

    print("best parameters:")
    print(dt_random.best_params_)

    # Evaluate
    best_random = dt_random.best_estimator_
    y_hat = best_random.predict(x_test)
    accuracy = accuracy_score(y_hat, y_test)

    print("Accuracy on test: {}%".format(round(accuracy, 6) * 100))

    # Return model for later usage
    return best_random

if __name__ == "__main__":
     main()