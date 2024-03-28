import logging
import pandas as pd
import os
import json
from datetime import datetime
from utils import load_config, _compute_indices
from bandpass_audio_retrieval import filtered_audio
import glob
from multiprocessing import Pool
from date_parser import parse_date

logger = logging.getLogger(__name__)


ref_1 = datetime.strptime("5:30", "%H:%M")
ref_2 = datetime.strptime("9:00", "%H:%M")
ref_3 = datetime.strptime("17:30", "%H:%M")
ref_4 = datetime.strptime("21:00", "%H:%M")
ref_5 = datetime.strptime("23:59", "%H:%M")

def region_filter_bp(df, region):
    config, path_data, last_dataset, path_exp, _, _, _, _, _ = load_config()
    region_int = int(region.split()[2])
    #region_dir = f'{region}'
    df['Date Time'] = pd.to_datetime(df['Date'])

    if region_int in [1, 2, 3, 4]:
        filtered_df = df[(df['Date Time'].dt.time  < ref_1.time())]
    elif region_int in [5, 6, 7, 8]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_1.time()) & (df['Date Time'].dt.time <= ref_2.time())]
    elif region_int in [9, 10, 11, 12]:
        filtered_df = df[(df['Date Time'].dt.time > ref_2.time()) & (df['Date Time'].dt.time <= ref_3.time())]
    elif region_int in [13, 14, 15, 16]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_3.time()) & (df['Date Time'].dt.time < ref_4.time())]
    elif region_int in [17, 18, 19, 20]:
        filtered_df = df[(df['Date Time'].dt.time >= ref_4.time()) & (df['Date Time'].dt.time <= ref_5.time())]

    files = []
    if os.path.exists(os.path.join(path_exp, last_dataset, region, 'acoustic_indices.csv')):
        logger.info(f'Acoustic indices already calculated for {region}')
        df_indices = pd.read_csv(os.path.join(path_exp, last_dataset, region, 'acoustic_indices.csv'), index_col=0)
        return df_indices

    elif os.path.exists(os.path.join(path_data, last_dataset, region)):
        if len(os.listdir(os.path.join(path_data, last_dataset, region))) == len(filtered_df):
            logger.info(f'All audios already bandpass filtered for region {region}')
            list_wav = glob.glob(os.path.join(path_data, last_dataset, region, '**', '*.wav'), recursive=True)

    else:
        for file in filtered_df.index:
            files.append(os.path.join(path_data, last_dataset, file))
        # This function already saves the filtered audios in a location
        filtered_audio(files, region)
        list_wav = glob.glob(os.path.join(path_data, last_dataset, region, '**', '*.wav'), recursive=True)

    logger.info(f'Computing acoustic indices for region {region}')
    with Pool() as pool:
        results = pool.map(_compute_indices, list_wav)
    filtered_results = [result for result in results if result is not None]
    corrupt_audio_paths = [audio_path for result, audio_path in zip(results, list_wav) if result is None]
    if filtered_results:
        df_indices = pd.concat(filtered_results)
        df_indices.index = pd.Index([os.path.basename(x) for x in list_wav if x not in corrupt_audio_paths])
        df_indices_date = parse_date(df_indices)
        if not os.path.exists(os.path.join(path_exp, last_dataset, region)):
            os.makedirs(os.path.join(path_exp, last_dataset, region))
        df_indices_date.to_csv(os.path.join(path_exp, last_dataset, region, 'acoustic_indices.csv'))
        return df_indices_date





