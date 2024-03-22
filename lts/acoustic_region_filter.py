import json
import pandas as pd
import os
from datetime import datetime
import ast

config = json.load(open('config.json'))

path_data = config["PATH_DATA"]
last_dataset = config["last_dataset"]


ref_1 = datetime.strptime("5:30", "%H:%M")
ref_2 = datetime.strptime("9:00", "%H:%M")
ref_3 = datetime.strptime("17:30", "%H:%M")
ref_4 = datetime.strptime("21:00", "%H:%M")
ref_5 = datetime.strptime("23:59", "%H:%M")

def region_filter(df, region):
    region = int(region.split()[2])
    df['Date Time'] = pd.to_datetime(df['Date'])

    if region in [1, 2, 3, 4]:
        filtered_df = df[(df['Date Time'].dt.time  < ref_1.time())]
    elif region in [5, 6, 7, 8]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_1.time()) & (df['Date Time'].dt.time <= ref_2.time())]
    elif region in [9, 10, 11, 12]:
        filtered_df = df[(df['Date Time'].dt.time > ref_2.time()) & (df['Date Time'].dt.time <= ref_3.time())]
    elif region in [13, 14, 15, 16]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_3.time()) & (df['Date Time'].dt.time < ref_4.time())]
    elif region in [17, 18, 19, 20]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_4.time()) & (df['Date Time'].dt.time <= ref_5.time())]

    filtered_df['File Name'] = filtered_df.index
    filtered_df = filtered_df.reset_index(drop=True)
    freq_ranges = {
        1: (0, 988), 5: (988, 3609), 9: (3609, 7906), 13: (7906, 11000),
        2: (988, 3609), 6: (988, 3609), 10: (988, 3609), 14: (988, 3609), 18: (988, 3609),
        3: (3609, 7906), 7: (3609, 7906), 11: (3609, 7906), 15: (3609, 7906), 19: (3609, 7906),
        4: (7906, 11000), 8: (7906, 11000), 12: (7906, 11000), 16: (7906, 11000), 20: (7906, 11000)
    }
    freq_min, freq_max = freq_ranges[region]

    indexes = []
    # Using ast.literal_eval because the frequencies list had type string
    for i, value in enumerate(ast.literal_eval(filtered_df['frequencies'].iloc[0])):
        if value >= freq_min and value < freq_max:
            indexes.append(i)

    # Filtering columns that have per_bin values
    selected_cols = [col for col in filtered_df.columns if 'per_bin' in col or 'Date Time' in col or 'File Name' in col]
    selected_cols.remove('AGI_per_bin')
    filtered_df = filtered_df[selected_cols]
    new_filtered_df = {}
    subset_indices = [[] for _ in range(len(filtered_df))]

    for col in selected_cols:
        if col != 'Date Time' and col != 'File Name':
            subset_indices = [[] for _ in range(len(filtered_df))]
            for x, value in enumerate(filtered_df[col]):
                index_values = []
                for i, val in enumerate(ast.literal_eval(value)):
                    if i in indexes:
                        index_values.append(val)
                subset_indices[x] = index_values
            new_filtered_df[col] = subset_indices
        elif col == 'Date Time':
            new_filtered_df['Date'] = filtered_df[col]
        elif col == 'File Name':
            new_filtered_df['File Name'] = filtered_df[col]


    new_filtered_df = pd.DataFrame(new_filtered_df)
    selected_cols.remove('Date Time')
    selected_cols.remove('File Name')
    for col in selected_cols:
        for x in range(len(new_filtered_df)):
            val = new_filtered_df.loc[x, col]
            value = sum(val) / len(val)
            new_filtered_df.loc[x, col] = value
    new_filtered_df.set_index('File Name', inplace=True)

    return new_filtered_df, selected_cols


