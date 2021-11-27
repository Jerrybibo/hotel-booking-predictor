import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split

# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE WRITTEN BY OTHER STUDENTS. Mimi Olayeye

def feature_reduction(datafile):
    # dropping irrelevant/redundant features
    datafile = datafile.drop(columns=['credit_card'])
    datafile = datafile.drop(columns=['phone-number'])
    datafile = datafile.drop(columns=['email'])
    datafile = datafile.drop(columns=['arrival_date_year'])
    datafile = datafile.drop(columns=['meal'])
    datafile = datafile.drop(columns=['country'])
    datafile = datafile.drop(columns=['market_segment'])
    datafile = datafile.drop(columns=['distribution_channel'])
    datafile = datafile.drop(columns=['company'])
    datafile = datafile.drop(columns=['reservation_status_date'])
    datafile = datafile.drop(columns=['name'])
    datafile = datafile.drop(columns=['arrival_date_month'])
    datafile = datafile.drop(columns=['reserved_room_type'])
    datafile = datafile.drop(columns=['assigned_room_type'])
    datafile = datafile.drop(columns=['reservation_status'])
    datafile = datafile.drop(columns=['agent'])
    datafile =datafile.dropna()

    # int values for categorical data:
    i = 0
    for hoteltype in datafile['hotel']:
        x = hoteltype[0]
        if hoteltype == 'Resort Hotel':
            datafile['hotel'][i] = 0
        if hoteltype == 'City Hotel':
            datafile['hotel'][i] = 1
        i = i+1
    i = 0
    for deposittype in datafile['deposit_type']:

        if deposittype == 'No Deposit':
            datafile['deposit_type'][i] = 0
        if deposittype == 'Non Refund':
            datafile['deposit_type'][i] = 1
        if deposittype == 'Refundable':
            datafile['deposit_type'][i] = 2
        i = i + 1
    i = 0
    for customertype in datafile['customer_type']:
        if customertype == 'Transient':
            datafile['customer_type'][i] = 0
        if customertype == 'Transient-Party':
            datafile['customer_type'][i] = 1
        if customertype == 'Group':
            datafile['customer_type'][i] = 2
        if customertype == 'Contract':
            datafile['customer_type'][i] = 3
        i = i + 1
    return datafile

def main():
    # set up the program to take in arguments from the command line

    # loading hotel booking dataframe
    hotel_booking = pd.read_csv("hotel_booking.csv",dtype = 'unicode')
    hotel_booking_2 = feature_reduction(hotel_booking)

    # hotel booking 2 has all irrelavant feature removed
    # & all categorical data is converted to numerical values
    hotel_booking_2= pd.DataFrame(hotel_booking_2)

    # outputing hotel 2
    hotel_booking_2.to_csv("hotel_booking_2.csv", index=False)










if __name__ == "__main__":
    main()