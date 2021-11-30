import pandas as pd
from numpy import ravel
from globals import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def process_dt(x_train, x_test, y_train, y_test):
    # Initialize decision tree classifier
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x_train, y_train)

    # Predict accuracy using KFold
    kf = KFold(n_splits=FOLD_COUNT, random_state=RANDOM_STATE, shuffle=True)
    kf_accuracies = []
    for train_i, test_i in kf.split(x_train):
        kf_x_train, kf_x_test = x_train[train_i], x_train[test_i]
        kf_y_train, kf_y_test = y_train[train_i], y_train[test_i]
        dt = dt.fit(kf_x_train, ravel(kf_y_train))
        kf_y_hat = dt.predict(kf_x_test)
        kf_accuracies.append(accuracy_score(kf_y_hat, kf_y_test))

    print("Accuracies of the {}-fold validation: {}".format(FOLD_COUNT,
                                                            list(map(lambda acc: round(acc * 100, 4), kf_accuracies))))
    print("Average accuracy prediction: {}%".format(round(sum(kf_accuracies) / len(kf_accuracies) * 100, 4)))

    # Refit model to full training set
    dt = dt.fit(x_train, ravel(y_train))

    # Calculate accuracy on test set
    y_hat = dt.predict(x_test)
    accuracy = accuracy_score(y_hat, y_test)
    print("Accuracy on test: {}%".format(round(accuracy, 6) * 100))

    # Return model for later usage
    return dt

# def main():
#     # Read dataset
#     x_train = pd.read_csv('../x_train.csv').to_numpy()
#     y_train = pd.read_csv('../y_train.csv').to_numpy()
#     x_test = pd.read_csv('../x_test.csv').to_numpy()
#     y_test = pd.read_csv('../y_test.csv').to_numpy()
#
#     # Initialize random forest classifier
#     dt = DecisionTreeClassifier()
#     # grid:
#     #number of trees
#     splitter = ["best", "random"]
#     #number of depth
#     max_depth = [10, 20, 50, 100, 200]
#     #msl
#     min_samples_leaf = [1, 5, 10, 20, 30]
#     #weight fraction leaf
#     min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]
#     #max_features
#     max_features = ["auto", "log2", "sqrt"]
#     #max_leaf_nodes
#     max_leaf_nodes = [10, 20, 30, 40, 50]
#
#     #create grid
#     random_grid = {'splitter': splitter,
#                    'max_features': max_features,
#                    'max_depth': max_depth,
#                    'min_samples_leaf': min_samples_leaf,
#                    'max_leaf_nodes': max_leaf_nodes,
#                    'min_weight_fraction_leaf': min_weight_fraction_leaf}
#
#     dt_random = RandomizedSearchCV(estimator=dt, param_distributions=random_grid,
#                                    n_iter=100, cv=3, verbose=2, random_state=42,
#                                    n_jobs=-1)
#
#     dt_random.fit(x_train, ravel(y_train))
#
#     print("best parameters:")
#     print(dt_random.best_params_)
#
#     # Evaluate
#     best_random = dt_random.best_estimator_
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