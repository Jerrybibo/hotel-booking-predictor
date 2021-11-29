import argparse
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from globals import *

# Silence chained assignment warnings from pandas
pd.options.mode.chained_assignment = None


def df_to_train_test(feature_df):
    # Splits our feature dataframe into train-test sets and organize them accordingly
    train, test = train_test_split(feature_df, shuffle=True, random_state=RANDOM_STATE)
    np_train, np_test = train.to_numpy(), test.to_numpy()
    y_train = pd.DataFrame(data=[int(s[0]) for s in np_train], columns=None)
    y_test = pd.DataFrame(data=[int(s[0]) for s in np_test], columns=None)
    x_train = pd.DataFrame(train)
    x_train.drop(columns=['is_canceled'], inplace=True)
    x_test = pd.DataFrame(test)
    x_test.drop(columns=['is_canceled'], inplace=True)
    return x_train, x_test, y_train, y_test


def feature_reduction(feature_df):
    # First, drop all irrelevant features
    reduced_df = feature_df[RELEVANT_FEATURES]

    # Remove all N/A entries and reset index
    reduced_df = reduced_df.dropna().reset_index(drop=True)

    # Feature modification
    # Combine arrival_date_month and arrival_date_day_of_month to form arrival_day_of_year
    # Also check if reserved room type matches assigned room type to reduce total iteration count (1 = matches)
    reduced_df['arrival_date_month'] = list(map(lambda x: MONTHS.index(x), reduced_df['arrival_date_month']))
    for i in range(len(reduced_df)):
        reduced_df['arrival_date_day_of_month'][i] = datetime(
            year=reduced_df['arrival_date_year'][i],
            month=reduced_df['arrival_date_month'][i] + 1,
            day=reduced_df['arrival_date_day_of_month'][i]
        ).timetuple().tm_yday
        reduced_df['reserved_room_type'][i] = [0, 1].index(
            (reduced_df['reserved_room_type'][i] == reduced_df['assigned_room_type'][i])
        )
    reduced_df.rename(columns={'arrival_date_day_of_month': 'arrival_day_of_year',
                               'reserved_room_type': 'room_request_matched'}, inplace=True)
    reduced_df.drop(columns=['arrival_date_month', 'arrival_date_year', 'assigned_room_type'], inplace=True)

    # Combine head count of children and babies to form minors count
    reduced_df['children'] = reduced_df['children'].add(reduced_df['babies']).astype(int)
    reduced_df.rename(columns={'children': 'minors'}, inplace=True)
    reduced_df.drop(columns=['babies'], inplace=True)

    # Check if the hotel booker is from Portugal (1 = not from Portugal)
    reduced_df['country'] = reduced_df['country'].apply(lambda x: [0, 1].index(x != 'PRT'))
    reduced_df.rename(columns={'country': 'is_foreign'}, inplace=True)

    # One-hot encoding for hotel, deposit_type and customer_type
    hotel_dummies = pd.get_dummies(reduced_df['hotel']).rename(columns={
        'City Hotel': 'city_hotel',
        'Resort Hotel': 'resort_hotel'})
    deposit_type_dummies = pd.get_dummies(reduced_df['deposit_type']).rename(columns={
        'No Deposit': 'no_deposit',
        'Non Refund': 'non_refund',
        'Refundable': 'refundable'
    })
    customer_type_dummies = pd.get_dummies(reduced_df['customer_type']).rename(columns={
        'Transient': 'transient',
        'Transient-Party': 'transient_party',
        'Contract': 'contract',
        'Group': 'group'
    })
    reduced_df.drop(columns=['hotel', 'deposit_type', 'customer_type'], inplace=True)
    reduced_df = pd.concat([reduced_df, hotel_dummies, deposit_type_dummies, customer_type_dummies], axis='columns')

    return reduced_df


def preprocess(x_train, x_test, y_train, y_test):
    # Given the four train-test sets, scale reasonable features
    std_scale_features = [
        'lead_time',
        'stays_in_weekend_nights',
        'stays_in_week_nights',
        'adults',
        'minors',
        'previous_cancellations',
        'previous_bookings_not_canceled',
        'booking_changes',
        'days_in_waiting_list',
        'adr',
        'total_of_special_requests'
    ]
    # Initialize StandardScaler and scale features
    std_scaler = StandardScaler()
    x_train[std_scale_features] = std_scaler.fit_transform(x_train[std_scale_features])
    x_test[std_scale_features] = std_scaler.transform(x_test[std_scale_features])
    return x_train, x_test, y_train, y_test


def preprocess_file(filename, out_file):
    start_time = time()
    print("Reading original dataset...")
    original_dataset = pd.read_csv(filename)
    print("Performing feature reduction...")
    reduced_dataset = feature_reduction(original_dataset)
    print("Exporting new dataset to {}...".format(out_file))
    reduced_dataset.to_csv(out_file, index=False)
    print("Splitting dataset to train-test sets...")
    x_train, x_test, y_train, y_test = df_to_train_test(reduced_dataset)
    print("Scaling the train-test feature sets...")
    x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test)
    print("Exporting the train-test files...")
    x_train.to_csv('x_train.csv', index=False)
    x_test.to_csv('x_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    print("Completed in {}s.".format(round(time() - start_time, 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Processes and train-test splits the input data into features and labels.")
    parser.add_argument("-i",
                        default="hotel_booking.csv",
                        help="filename of the input data")
    parser.add_argument("-o",
                        default="hotel_booking_processed.csv",
                        help="filename of the output (processed) data")
    args = parser.parse_args()
    preprocess_file(args.i, args.o)


if __name__ == "__main__":
    main()
