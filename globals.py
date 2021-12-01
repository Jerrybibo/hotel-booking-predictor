RANDOM_STATE = 334

FOLD_COUNT = 10

USE_LEGACY_DATASET = False

PICKLE_MODELS = False

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

RELEVANT_FEATURES = [
    "is_canceled",  # Label
    "hotel",        # Needs to be 1-hot encoded into (city_hotel, resort_hotel)
    "lead_time",
    "arrival_date_year",            # See below (year used for leap year calculation)
    "arrival_date_month",           # See below
    "arrival_date_day_of_month",    # Combined with above to form arrival_day_of_year
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",     # See below
    "babies",       # Combined with above to form minors
    "country",      # Check against PRT -> (1, 0) to form foreign_traveler
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",   # See below
    "assigned_room_type",   # Compared with above to form room_request_matched
    "booking_changes",
    "deposit_type",     # Needs to be 1-hot encoded into (no_deposit, non_refund, refundable)
    "days_in_waiting_list",
    "customer_type",    # Needs to be 1-hot encoded into (transient_party, transient, contract, group)
    "adr",
    "total_of_special_requests"
]

STD_SCALE_FEATURES = [
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

FINAL_FEATURES = [
    "lead_time",
    "arrival_day_of_year",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "minors",
    "is_foreign",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "room_request_matched",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
    "city_hotel",
    "resort_hotel",
    "no_deposit",
    "non_refund",
    "refundable",
    "contract",
    "group",
    "transient",
    "transient_party"
]

DEFAULT_FEATURE_VALUES = {
    "lead_time": 0,
    "arrival_day_of_year": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 1,
    "adults": 2,
    "minors": 0,
    "is_foreign": 1,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "room_request_matched": 1,
    "booking_changes": 0,
    "days_in_waiting_list": 0,
    "adr": 100,
    "total_of_special_requests": 0,
    "city_hotel": 1,
    "resort_hotel": 0,
    "no_deposit": 1,
    "non_refund": 0,
    "refundable": 0,
    "contract": 0,
    "group": 0,
    "transient": 0,
    "transient_party": 1
}
