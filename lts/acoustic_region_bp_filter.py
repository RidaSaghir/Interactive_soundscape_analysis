import logging
import pandas as pd
import os
import json
from datetime import datetime
from utils import load_config, _compute_indices
from bandpass_audio_retrieval import filter_audio
import glob
from multiprocessing import Pool
from date_parser import parse_date
from audio_file_trimmer import trimmer

logger = logging.getLogger(__name__)


ref_1 = datetime.strptime("5:30", "%H:%M")
ref_2 = datetime.strptime("9:00", "%H:%M")
ref_3 = datetime.strptime("17:30", "%H:%M")
ref_4 = datetime.strptime("21:00", "%H:%M")
ref_5 = datetime.strptime("23:59", "%H:%M")

def region_filter_bp(df):
    """
        Filters the data frame based on chosen region. It performs time filtering and bandpass filtering

        Args:
            df : Data frame having series index, features (indices or VAE), 'Date Time' and 'File Name' columns.
        Returns:
            df_with_dates : Same format data frame but region filtered.
        """
    config, path_data, last_dataset, path_exp, clustering_rep, resolution, _, _, _, acoustic_region = load_config()
    region_int = int(acoustic_region.split()[2])
    df['Date Time'] = pd.to_datetime(df['Date Time'])

    files = []
    if os.path.exists(os.path.join(path_exp, last_dataset, acoustic_region, f'{clustering_rep}_{resolution}.csv')):
        logger.info(f'Acoustic indices already calculated for {acoustic_region}')
        df_indices = pd.read_csv(os.path.join(path_exp, last_dataset, acoustic_region, f'{clustering_rep}_{resolution}.csv'), index_col=0)
        return df_indices

    else:
        if region_int in [1, 2, 3, 4]:
            filtered_df = df[(df['Date Time'].dt.time < ref_1.time())]
        elif region_int in [5, 6, 7, 8]:
            filtered_df = df[(df['Date Time'].dt.time >= ref_1.time()) & (df['Date Time'].dt.time <= ref_2.time())]
        elif region_int in [9, 10, 11, 12]:
            filtered_df = df[(df['Date Time'].dt.time > ref_2.time()) & (df['Date Time'].dt.time <= ref_3.time())]
        elif region_int in [13, 14, 15, 16]:
            filtered_df = df[(df['Date Time'].dt.time >= ref_3.time()) & (df['Date Time'].dt.time < ref_4.time())]
        elif region_int in [17, 18, 19, 20]:
            filtered_df = df[(df['Date Time'].dt.time >= ref_4.time()) & (df['Date Time'].dt.time <= ref_5.time())]

        for file in filtered_df['File Name']:
            files.append(os.path.join(path_data, last_dataset, file))

        filtered_audios = filter_audio(files, acoustic_region)
        df_indices = pd.DataFrame()
        for audio in filtered_audios:
            indices_df = _compute_indices(audio)
            df_indices = pd.concat([df_indices, indices_df], axis=0)
        df_indices.index = filtered_df['File Name']
        df_indices_date = parse_date(df_indices)
        return df_indices_date








