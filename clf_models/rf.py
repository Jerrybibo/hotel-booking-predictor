from numpy import ravel
from globals import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


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

