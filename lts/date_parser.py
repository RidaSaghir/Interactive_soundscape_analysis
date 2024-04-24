import re
import json
from datetime import datetime
from multiprocessing import Pool
import pandas as pd


def parse_date(df):
    """
    Parses the date and rearranges the data frame suitable for next operations

    Args:
        df : Original data frame returned from MAAD scikit library for computing indices with index as file names.
    Returns:
        df_with_dates : Data frame having additional column for 'Date Time' and 'File Name' with series index.
    """
    with Pool() as pool:
        results = pool.map(_parse_date, df.iterrows())

    df_with_dates = pd.concat(results, axis=0)
    df_with_dates.reset_index(drop=True, inplace=True)
    df_with_dates.fillna(0, inplace=True)
    # Deletes the following columns as they often result in infinity values
    df_with_dates.drop(['AGI', 'AGI_per_bin'], axis=1, inplace=True)
    return df_with_dates

def _parse_date(row):
    date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'
    index, series = row
    match = re.search(date_time_pattern, index)
    if match:
        year, month, day, hour, minute, second = match.groups()
        timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        series['Date Time'] = timestamp

        # Create a DataFrame with a single row from the modified Series
    df_row = pd.DataFrame(series).transpose()
    #df_row['File Name'] = df_row.index
    return df_row

def parse_date_snippets(snippets):
    """
        Parses the date from audio snippets
        Args:
            snippets : a list of tuples having (file name, audio data, frame rate). The whole list of snippets would belong to a single file.
        Returns:
            results : a list of tuples having (timestamp, file name, audio data, frame rate)
    """
    config = json.load(open('config.json'))
    resolution = config["resolution"]
    result = re.search(r'\d+', resolution)
    resolution = int(result.group())
    date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'
    results = []
    for i, snippet in enumerate(snippets):
        file_name, s, fs = snippet
        match = re.search(date_time_pattern, file_name)
        if match:
            year, month, day, hour, minute, second = match.groups()
            second = int(second) + (i * resolution)
            timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        else:
            timestamp = None
        result = (timestamp, file_name, s, fs)
        results.append(result)
    return results
