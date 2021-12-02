# For usage in web server. See more at http://jerry.games/334

import sys
import base64
import json
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from time import time
from datetime import date, timedelta
import random
from globals import *

# Obtain the input and convert to dict
arg = sys.argv[1]
raw_input_row = json.loads(base64.b64decode(arg))

start_time = time()

# Unpickle our classifier of choice
classifier_name = raw_input_row['classifier']
# Todo fix directory
with open('../pickled_models/{}.pickle'.format(classifier_name), 'rb') as f:
    classifier = pickle.load(f)

# Preprocess our input data to match that requested by the models
input_row = deepcopy(DEFAULT_FEATURE_VALUES)

input_row['lead_time'] = int(raw_input_row['lead_time'])
# arrival_date -> arrival_day_of_year
month, day, year = map(int, raw_input_row['arrival_date'].split('/'))
input_row['arrival_day_of_year'] = date(year, month, day).timetuple().tm_yday
# approximate weekend and week nights from stay duration
input_row['stays_in_week_nights'] = np.busday_count(date(year, month, day), date(year, month, day) +
                                                 timedelta(days=int(raw_input_row['stay_duration'])))
input_row['stays_in_weekend_nights'] = int(raw_input_row['stay_duration']) - input_row['stays_in_week_nights']
input_row['adults'] = int(raw_input_row['adults'])
input_row['minors'] = int(raw_input_row['minors'])
input_row['is_foreign'] = ['off', 'on'].index(raw_input_row['is_foreign'])
for hotel_type in ['city_hotel', 'resort_hotel']:
    input_row[hotel_type] = 1 if hotel_type == raw_input_row['hotel_type'] else 0
for deposit_type in ['no_deposit', 'non_refund', 'refundable']:
    input_row[deposit_type] = 1 if deposit_type == raw_input_row['deposit_type'] else 0
for customer_type in ['transient', 'group', 'transient_party', 'contract']:
    input_row[customer_type] = 1 if customer_type == raw_input_row['customer_type'] else 0

# Todo fix directory
with open('../pickled_models/std_scaler.pickle', 'rb') as f:
    std_scaler = pickle.load(f)
input_row = pd.DataFrame.from_dict([input_row])
input_row[STD_SCALE_FEATURES] = std_scaler.transform(input_row[STD_SCALE_FEATURES])

input_row = input_row.to_numpy()

result = dict()
if classifier_name in ['lgr', 'rf']:
    result['will_cancel'] = str(round(classifier.predict_proba(input_row)[0][1] * 100, 1)) + "%"
else:
    result['will_cancel'] = str(classifier.predict(input_row)[0])
result['time_elapsed'] = round(time() - start_time, 2)

print(result)