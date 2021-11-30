import numpy as np
import pandas as pd
from numpy import ravel
from globals import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


def process_rf(x_train, x_test, y_train, y_test):
    # Initialize random forest classifier
    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt').fit(x_train, ravel(y_train))

    # Predict accuracy using KFold
    kf = KFold(n_splits=FOLD_COUNT, random_state=RANDOM_STATE, shuffle=True)
    kf_accuracies = []
    for train_i, test_i in kf.split(x_train):
        kf_x_train, kf_x_test = x_train[train_i], x_train[test_i]
        kf_y_train, kf_y_test = y_train[train_i], y_train[test_i]
        rf = rf.fit(kf_x_train, ravel(kf_y_train))
        kf_y_hat = rf.predict(kf_x_test)
        kf_accuracies.append(accuracy_score(kf_y_hat, kf_y_test))

    print("Accuracies of the {}-fold validation: {}".format(FOLD_COUNT,
                                                            list(map(lambda acc: round(acc * 100, 4), kf_accuracies))))
    print("Average accuracy prediction: {}%".format(round(sum(kf_accuracies) / len(kf_accuracies) * 100, 4)))

    # Refit model to full training set
    rf = rf.fit(x_train, ravel(y_train))

    # Calculate accuracy on test set
    y_hat = rf.predict(x_test)
    accuracy = accuracy_score(y_hat, y_test)
    print("Accuracy on test: {}%".format(round(accuracy, 6) * 100))

    # Return model for later usage
    return rf

# def main():
#     # Read dataset
#     x_train = pd.read_csv('../x_train.csv').to_numpy()
#     y_train = pd.read_csv('../y_train.csv').to_numpy()
#     x_test = pd.read_csv('../x_test.csv').to_numpy()
#     y_test = pd.read_csv('../y_test.csv').to_numpy()
#
#     # Initialize random forest classifier
#     rf = RandomForestClassifier()
#     # grid:
#     #number of trees
#     n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
#     #max features
#     max_features = ['auto', 'sqrt']
#     #number of depth
#     max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
#     max_depth.append(None)
#     #min sample required for split
#     min_samples_split = [50, 100, 150]
#     #msl
#     min_samples_leaf = [1, 10, 20]
#     #bootstrap
#     bootstrap = [True, False]
#
#     #create grid
#     random_grid = {'n_estimators': n_estimators,
#                    'max_features': max_features,
#                    'max_depth': max_depth,
#                    'min_samples_split': min_samples_split,
#                    'min_samples_leaf': min_samples_leaf,
#                    'bootstrap': bootstrap}
#
#     rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
#                                    n_iter=100, cv=3, verbose=2, random_state=42,
#                                    n_jobs=-1)
#
#     rf_random.fit(x_train, ravel(y_train))
#
#     print("best parameters:")
#     print(rf_random.best_params_)
#
#     # Evaluate
#     best_random = rf_random.best_estimator_
#     y_hat = best_random.predict(x_test)
#     accuracy = accuracy_score(y_hat, y_test)
#
#     print("Accuracy on test: {}%".format(round(accuracy, 6) * 100))
#
#     # Return model for later usage
#     return best_random
#
# if __name__ == "__main__":
#      main()
