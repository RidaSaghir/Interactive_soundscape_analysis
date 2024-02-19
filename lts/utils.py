import numpy as np
from aci_in_region import compute_ACI
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
# Construct the absolute path to the config.json file
config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
# Load the config.json file
with open(config_file_path) as config_file:
    config = json.load(config_file)
#config = json.load(open('config.json'))
PATH_DATA = config.get('PATH_DATA')
PATH_EXP = os.path.join(os.path.dirname(PATH_DATA), 'exp')
logging.debug('PATH_DATA: {}'.format(PATH_DATA))

def list_datasets():
    ds_choices = [i for i in os.listdir(PATH_DATA) if os.path.isdir(os.path.join(PATH_DATA, i))]
    ds_value = json.load(open(config_file_path)).get('last_dataset')
    ds_value = ds_value if ds_value in ds_choices else None
    return ds_choices, ds_value


def update_last_dataset(datset):
    config['last_dataset'] = datset
    with open('config.json', 'w') as f:
        json.dump(config, f)

def summarise_dataset(dataset):
    # TODO Find out why it takes so long (>10s) to run this bit. Make it faster.
    list_wav = glob.glob(os.path.join(PATH_DATA, dataset, '**', '*.wav'), recursive=True)
    is_computed = os.path.isfile(os.path.join(PATH_EXP, dataset, 'all_indices.csv'))
    m = f'{len(list_wav)} files found.\nIndices already computed: {is_computed}'

    logger.debug(f'Counting total size of {len(list_wav)} wav files')
    with Pool() as pool:
        results = pool.map(os.path.getsize, list_wav)
    m += f'\nTotal size: {sum(results) / 1024 ** 3:0.2f} GB'
    logger.debug('Done')
    return m

def compute_indices(dataset, btn_compute):
    list_wav = glob.glob(os.path.join(PATH_DATA, dataset, '**', '*.wav'), recursive=True)
    is_computed = os.path.isfile(os.path.join(PATH_EXP, dataset, 'all_indices.csv'))

    if not is_computed or btn_compute == 'Recompute indices?':
        # TODO In case is_computed is True, ask whether to recompute
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
            df_indices_date.to_csv(os.path.join(PATH_EXP, dataset, 'all_indices.csv'))
            logger.info('Done')
            message = f'Results saved to csv.'
        else:
            logger.warning("No valid results to save.")
            message = f'No valid results to save.'
            # Optionally, handle the case where no valid results are obtained.

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
        # No normalized values for ACI in maad. That's why doing manually
        all_indices['ACI_normalized'] = (all_indices['ACI'] - all_indices['ACI'].min()) / (all_indices['ACI'].max() - all_indices['ACI'].min())

        return all_indices

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
    return None


def calculate_values(folder_path, audio_file, hour, minute, timestamp):

    audio_path = os.path.join(folder_path, audio_file)  # Create the full path to the audio file
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    w_filtered = maad.sound.select_bandwidth(y, sr, fcut=(482, 12000), forder=5,
                                             fname='butter',
                                             ftype='bandpass')
    #Sxx, freq = compute_spectrogram(sr, y)
    Sxx, tn, fn, ext = maad.sound.spectrogram(y, sr, window='hann', nperseg=512, noverlap=0)
    # Background Noise (BGN)
    Sxx_dB = maad.util.power2dB(Sxx) + 96
    #Sxx_dB_noNoise, noise_profile, _ = maad.sound.remove_background(Sxx_dB)
    Sxx_dB_noNoise_med, noise_profile = maad.sound.remove_background_along_axis(Sxx_dB, mode='median')
    # Signal to noise ratio (It already does power2db and median filtering)
    _, _, snr, _, _, _ = maad.sound.spectral_snr(Sxx)
    # Event Count Index (it by defaults takes in waveform decibels)
    EVNtFract, EVNmean, EVNcount, _ = maad.features.temporal_events(w_filtered, sr, 3)
    EVN = np.mean(EVNtFract)
    # Spectral Activity (ACT) with noise reduced decibel envelope
    ACTspfracT, ACTspcount, ACTspmean = maad.features.spectral_activity(Sxx_dB_noNoise_med, dB_threshold=3)


    # Acoustic Cover index
    LFC, MFC, HFC = maad.features.spectral_cover(Sxx_dB_noNoise_med, fn, dB_threshold=2, flim_LF=(1, 1000),
                                                 flim_MF=(1000, 8000), flim_HF=(8000, 10982))

    # Entropy of peaks spectrum (EPS), Entropy of average spectrum (EAS), Entropy of spectrum of coefficients of variations (ECV)
    EAS, ECU, ECV, EPS, EPS_KURT, EPS_SKEW = maad.features.spectral_entropy(Sxx_dB_noNoise_med, fn, flim=(1000, 8000))
    # Acoustic complexity index
    _, aci_per_bin , aci_main = maad.features.acoustic_complexity_index(Sxx)
    min_ACI = min(aci_per_bin)
    max_ACI = max(aci_per_bin)

    # Normalize ACI values to [0, 1]
    normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
    aci = np.mean(normalized_ACI)

    # ACI in regions
    regions = {
        'Region 1': None,
        'Region 2': None,
        'Region 3': None,
        'Region 4': None,
        'Region 5': None,
        'Region 6': None,
        'Region 7': None,
        'Region 8': None,
        'Region 9': None,
        'Region 10': None,
        'Region 11': None,
        'Region 12': None,
        'Region 13': None,
        'Region 14': None,
        'Region 15': None,
        'Region 16': None,
        'Region 17': None,
        'Region 18': None,
        'Region 19': None,
        'Region 20': None
    }
    for region in regions:
        aci_region = compute_ACI(region, Sxx, fn, sr, timestamp)
        regions[region] = aci_region

    # Temporal Entropy Index
    Hf, Ht_per_bin = maad.features.frequency_entropy(Sxx)



    # Acoustic Evenness Index
    #AEI = maad.features.acoustic_eveness_index(Sxx, fn, fmax=11000, dB_threshold=-47)
    # Acoustic Diversity Index
    #ADI = maad.features.acoustic_diversity_index(Sxx, fn, fmax=10000, dB_threshold=-47)

    return noise_profile, snr, aci, Hf, EVN, LFC, MFC, HFC, ACTspcount, EPS, EAS, ECV, regions