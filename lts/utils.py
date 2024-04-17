import glob
import json
import time
import logging
import os
from multiprocessing import Pool
import maad
import numpy as np
import pandas as pd
from maad import sound, features
from date_parser import parse_date

logger = logging.getLogger(__name__)
# config = json.load(open('config.json'))
# PATH_DATA = config.get('PATH_DATA')
# PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')
# logging.debug('PATH_DATA: {}'.format(PATH_DATA))

def load_config():
    config = json.load(open('config.json'))
    path_data = config["PATH_DATA"]
    last_dataset = config["last_dataset"]
    path_exp = os.path.join(os.path.dirname(path_data), 'exp')
    clustering_rep = config["clustering_rep"]
    resolution = config["resolution"]
    clustering_mode = config["clustering_mode"]
    dim_red_mode = config["dim_red_mode"]
    clustering_filter = config["clustering_filter"]
    acoustic_region = config["acoustic_region"]
    return (config, path_data, last_dataset, path_exp, clustering_rep, resolution,
            clustering_mode, dim_red_mode, clustering_filter, acoustic_region)

def list_datasets():
    _, PATH_DATA, last_dataset, _, _, _, _, _, _, _ = load_config()
    ds_choices = [i for i in os.listdir(PATH_DATA) if os.path.isdir(os.path.join(PATH_DATA, i))]
    ds_value = last_dataset
    ds_value = ds_value if ds_value in ds_choices else None
    return ds_choices, ds_value


def update_last_dataset(dataset):
    config, _, _, _, _, _, _, _, _ = load_config()
    config['last_dataset'] = dataset
    with open('config.json', 'w') as f:
        json.dump(config, f)

def update_clustering_rep(rep):
    config, _, _, _, _, _, _, _, _ = load_config()
    config['clustering_rep'] = rep
    with open('config.json', 'w') as f:
        json.dump(config, f)

def update_clustering_mode(mode):
    config, _, _, _, _, _, _, _, _ = load_config()
    config['clustering_mode'] = mode
    with open('config.json', 'w') as f:
        json.dump(config, f)

def update_clustering_filter(filter):
    config, _, _, _, _, _, _, _, _ = load_config()
    if filter == 'none':
        config['acoustic_region'] = 'none'
    config['clustering_filter'] = filter
    with open('config.json', 'w') as f:
        json.dump(config, f)


def update_dim_red(red_mode):
    config, _, _, _, _, _, _, _, _ = load_config()
    config['dim_red_mode'] = red_mode
    with open('config.json', 'w') as f:
        json.dump(config, f)

def update_region(region):
    config, _, _, _, _, _, _, _, _ = load_config()
    config['acoustic_region'] = region
    with open('config.json', 'w') as f:
        json.dump(config, f)

def summarise_dataset(dataset):
    config, PATH_DATA, last_dataset, PATH_EXP, _, resolution, _, _, _, _ = load_config()
    # TODO Find out why it takes so long (>10s) to run this bit. Make it faster.
    list_wav = glob.glob(os.path.join(PATH_DATA, dataset, '**', '*.wav'), recursive=True)
    is_computed = os.path.isfile(os.path.join(PATH_EXP, dataset, 'acoustic_indices.csv'))
    m = f'{len(list_wav)} files found.\nIndices already computed: {is_computed}'

    logger.debug(f'Counting total size of {len(list_wav)} wav files')
    with Pool() as pool:
        results = pool.map(os.path.getsize, list_wav)
    m += f'\nTotal size: {sum(results) / 1024 ** 3:0.2f} GB'
    logger.debug('Done')
    return m

def compute_indices(dataset, btn_compute):
    #Todo: To compute indices based on different resolutions
    config, PATH_DATA, last_dataset, PATH_EXP, _, resolution, _, _, _, _ = load_config()
    list_wav = glob.glob(os.path.join(PATH_DATA, dataset, '**', '*.wav'), recursive=True)
    is_computed = os.path.isfile(os.path.join(PATH_EXP, dataset, f'acoustic_indices_{resolution}.csv'))

    if not is_computed or btn_compute == 'Recompute indices?':
        logger.info('Computing indices...')
        t_start = time.perf_counter()
        with Pool() as pool:
            results = pool.map(_compute_indices, list_wav)
        logger.info(f'Computation took {time.perf_counter() - t_start} seconds')
        # Filter out None results and collect corrupt audio paths
        t_start_filter = time.perf_counter()
        filtered_results = [result for result in results if result is not None]
        corrupt_audio_paths = [audio_path for result, audio_path in zip(results, list_wav) if result is None]
        logger.info(f'Filtering took {time.perf_counter() - t_start_filter} seconds')

        if filtered_results:
            df_indices = pd.concat(filtered_results)
            # Update index only if there are valid results
            df_indices.index = pd.Index([os.path.basename(x) for x in list_wav if x not in corrupt_audio_paths])
            df_indices_date = parse_date(df_indices)
            logger.info(f'Creating and saving CSV files')
            os.makedirs(os.path.join(PATH_EXP, dataset), exist_ok=True)
            #TODO: Do not hard code the name. Should vary with resolution
            df_indices_date.to_csv(os.path.join(PATH_EXP, dataset, 'acoustic_indices_1min.csv'))
            logger.info('Done')
            message = f'Results saved to csv.'
        else:
            logger.warning("No valid results to save.")
            message = f'No valid results to save.'

        if corrupt_audio_paths:
            message = f'Following files can not be processed: {corrupt_audio_paths}. Rest of the results are saved to csv. '

        return message

    else:
        message = f'Results of chosen folder already present. Do you want to recompute?'
        return message
def _compute_indices(audio_path):
    try:
        s, fs = maad.sound.load(audio_path, sr=None)
        temporal_indices = maad.features.all_temporal_alpha_indices(s, fs)

        Sxx_power, tn, fn, ext = maad.sound.spectrogram(s, fs)
        spectral_indices, per_bin_indices = maad.features.all_spectral_alpha_indices(Sxx_power, tn, fn)
        temp_spec = temporal_indices.join(spectral_indices)
        all_indices = temp_spec.join(per_bin_indices)
        return all_indices

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
    return None


def _compute_indices_without_wav(s, fs):
    temporal_indices = maad.features.all_temporal_alpha_indices(s, fs)
    Sxx_power, tn, fn, ext = maad.sound.spectrogram(s, fs)
    spectral_indices, per_bin_indices = maad.features.all_spectral_alpha_indices(Sxx_power, tn, fn)
    temp_spec = temporal_indices.join(spectral_indices)
    all_indices = temp_spec.join(per_bin_indices)
    return all_indices





