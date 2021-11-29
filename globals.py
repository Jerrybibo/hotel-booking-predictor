RANDOM_STATE = 334

FOLD_COUNT = 10

USE_LEGACY_DATASET = False

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
