# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Mimi Olayeye

import argparse
import pandas as pd
from globals import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def model_assessment(datafile):
    xtrain, xtest = train_test_split(datafile, shuffle=True, random_state=RANDOM_STATE)  # split

    train = xtrain.to_numpy()  # train with label "is_cancelled"
    test = xtest.to_numpy()  # test with label "is_cancelled"

    # initializing y (label) arrays
    ytrain = [int(s[0]) for s in train]
    ytest = [int(s[0]) for s in test]

    ytrain = pd.DataFrame(data=ytrain, columns=None)  # dataframe
    ytest = pd.DataFrame(data=ytest, columns=None)
    xtrain = pd.DataFrame(data=xtrain)  # dataframe
    xtest = pd.DataFrame(data=xtest)

    xtrain = xtrain.drop(columns=['is_canceled'])  # removing label
    xtest = xtest.drop(columns=['is_canceled'])  # removing label

    return xtrain, xtest, ytrain, ytest


def preprocessing(xTrain, xTest, yTrain, yTest):
    # Scale/Normalize the data:
    # Todo: Choose only continuous features to standardize, then reproduce hotel_booking_2.csv

    # one hot encoding
    # one_hot = OneHotEncoder(categorical_features=[0])

    # standard scaler
    standardized_scale = StandardScaler()

    xTrain[['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'booking_changes',
            'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
            'total_of_special_requests']] = standardized_scale.fit_transform(
        xTrain[['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
                'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'booking_changes',
                'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']])

    xTest[['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
           'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'booking_changes',
           'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
           'total_of_special_requests']] = standardized_scale.transform(
        xTest[['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights',
               'stays_in_week_nights', 'adults', 'children', 'babies', 'previous_cancellations', 'booking_changes',
               'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
               'total_of_special_requests']])  # transforming test

    # remove features using Pear Corr or manually (if necessary)

    # Run PCA to reduce dimensionality (95% variance)
    #
    # pca = PCA(n_components=0.95) # 95% variance XTRAIN
    # pca.fit(X=xTrain, y=yTrain)
    #
    # pca_test = PCA(n_components=0.95)  # 95% variance XTEST
    # pca_test.fit(X=xTest, y=yTest)
    #
    # xTrainPCA= pca.transform(X=xTrain)
    # xTestPCA= pca_test.transform(X=xTest)
    #
    #
    # yTrain = pd.DataFrame(data=yTrain, columns=None)  # dataframe
    # yTest = pd.DataFrame(data=yTest, columns=None)
    # xTrainPCA = pd.DataFrame(data=xTrainPCA)  # dataframe
    # xTestPCA = pd.DataFrame(data=xTestPCA)

    return xTrain, xTest, yTrain, yTest


def main():
    # -->  python preprocessing.py xTrain.csv xTest.csv yTrain.csv yTest.csv
    # set up the program to take in arguments from the command line

    parser = argparse.ArgumentParser()
    parser.add_argument("out_x_Train",
                        help="filename of the updated training data")
    parser.add_argument("out_x_Test",
                        help="filename of the updated test data")
    parser.add_argument("out_y_Train",
                        help="filename of the updated training data")
    parser.add_argument("out_y_Test",
                        help="filename of the updated test data")
    parser.add_argument("--data",
                        default="hotel_booking_2.csv",
                        help="filename of the input data")
    args = parser.parse_args()

    hotel_booking_2 = pd.read_csv("hotel_booking_2.csv", dtype='unicode')

    # split into x train/test and y train/test
    xtrain, xtest, ytrain, ytest = model_assessment(hotel_booking_2)

    # normalized and PCA
    xtrain, xtest, ytrain, ytest = preprocessing(xtrain, xtest, ytrain, ytest)

    # output
    xtrain.to_csv(args.out_x_Train, index=False)
    xtest.to_csv(args.out_x_Test, index=False)
    ytrain.to_csv(args.out_y_Train, index=False)
    ytest.to_csv(args.out_y_Test, index=False)


if __name__ == "__main__":
    main()
