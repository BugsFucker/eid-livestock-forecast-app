import datetime as dt

# Basic configuration
DATA_PATH = 'data/sheep_goat_prices_eu.csv'
MODEL_DIR = 'models'

# Example Eid al-Adha dates (Gregorian) - extend as needed
EID_DATES = {
    2015: dt.date(2015, 9, 24),
    2016: dt.date(2016, 9, 12),
    2017: dt.date(2017, 9, 1),
    2018: dt.date(2018, 8, 21),
    2019: dt.date(2019, 8, 11),
    2020: dt.date(2020, 7, 31),
    2021: dt.date(2021, 7, 20),
    2022: dt.date(2022, 7, 9),
    2023: dt.date(2023, 6, 28),
    2024: dt.date(2024, 6, 16),
    2025: dt.date(2025, 6, 6),
}
