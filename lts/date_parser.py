import re
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
def parse_date(df):
    with Pool() as pool:
        results = pool.map(_parse_date, df.iterrows())

    df_with_dates = pd.concat(results, axis=0)
    return df_with_dates

def _parse_date(row):
    date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'
    index, series = row
    match = re.search(date_time_pattern, index)
    if match:
        year, month, day, hour, minute, second = match.groups()
        timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        series['Date'] = timestamp

        # Create a DataFrame with a single row from the modified Series
    df_row = pd.DataFrame(series).transpose()

    return df_row