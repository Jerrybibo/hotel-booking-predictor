import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from globals import *


def feature_reduction(feature_df):
    # First, drop all irrelevant features
    reduced_df = feature_df[RELEVANT_FEATURES]

    # Remove all N/A entries and reset index
    reduced_df = reduced_df.dropna().reset_index()

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


def preprocess(feature_df):
    # todo
    # Given the now-reduced features dataframe, scale features reasonably
    return


def preprocess_file(filename, out_file):
    original_dataset = pd.read_csv(filename)
    reduced_dataset = feature_reduction(original_dataset)
    reduced_dataset.to_csv(out_file, index=False)
    preprocess(reduced_dataset)


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
