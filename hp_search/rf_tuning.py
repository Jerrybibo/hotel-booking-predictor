import numpy as np
import pandas as pd
from numpy import ravel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


def main():
    # Read dataset
    x_train = pd.read_csv('../x_train.csv').to_numpy()
    y_train = pd.read_csv('../y_train.csv').to_numpy()
    x_test = pd.read_csv('../x_test.csv').to_numpy()
    y_test = pd.read_csv('../y_test.csv').to_numpy()

    # Initialize random forest classifier
    rf = RandomForestClassifier()
    # grid:
    # number of trees
    n_estimators = list(range(30, 39))
    # max features
    max_features = ['auto', 'sqrt']
    # number of depth
    max_depth = list(range(20, 30))
    # msl
    min_samples_leaf = [1, 2, 3]
    # bootstrap
    bootstrap = [True, False]

    # create grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2, random_state=42,
                                   n_jobs=-1)

    rf_random.fit(x_train, ravel(y_train))

    print("best parameters:")
    print(rf_random.best_params_)

    # Evaluate
    best_random = rf_random.best_estimator_
    y_hat = best_random.predict(x_test)
    accuracy = accuracy_score(y_hat, y_test)

    print("Accuracy on test: {}%".format(round(accuracy, 6) * 100))

    # Return model for later usage
    return best_random


if __name__ == "__main__":
     main()
