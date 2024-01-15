import pandas as pd
import os
import re
import maad
import matplotlib.pyplot as plt

from maad.util import (
    date_parser, plot_correlation_map,
    plot_features_map, plot_features, false_Color_Spectro
    )
from maad import sound, features

indices = ['Ht_per_bin', 'ACI_per_bin', 'EVNspCount_per_bin']

def create_dataframe_from_wav_files(files):
    # List all files in the directory

    # Filter out only the WAV files
    wav_files = [file for file in files if file.endswith('.wav')]

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['file'], index=pd.to_datetime([]))

    # Define a regex pattern to match date and time in filenames
    date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'

    # Iterate through each WAV file
    for wav_file in wav_files:
        # Extract date and time using regex pattern
        match = re.search(date_time_pattern, wav_file)
        if match:
            year, month, day, hour, minute, second = match.groups()
            date_str = f'{year}{month}{day}'
            time_str = f'{hour}{minute}{second}'
            datetime_str = date_str + '_' + time_str
            #datetime_obj = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
            datetime_obj = pd.to_datetime(datetime_str, format='%Y%m%d_%H%M%S')

            # Create a new row in the DataFrame
            df.loc[datetime_obj] = os.path.join(os.path.dirname(files[0]), wav_file)


    # Sort the DataFrame based on the index (datetime)
    df = df.sort_index()

    return df

def calculate_indices(file, indices):

    try:
        s, fs = maad.sound.load(file)
    except Exception as e:
        print(f"Error loading audio file {file}: {e}")
        return None
    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=s,
        fs=fs,
        window='hann',
        nperseg=1024,
        noverlap=1024 // 2,
        verbose=False,
        display=False,
        savefig=None
    )
    Sxx_power, _, _, _ = maad.sound.spectrogram(s, fs)
    Sxx, _, _, _ = maad.sound.spectrogram(s, fs, mode='amplitude')
    Sxx_noNoise= maad.sound.median_equalizer(Sxx_power)
    Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)

    result = {}

    if 'ACI_per_bin' in indices:
        _, ACI_per_bin, _ = maad.features.acoustic_complexity_index(Sxx)
        result['ACI_per_bin'] = ACI_per_bin

    if 'Ht_per_bin' in indices:
        _, Ht_per_bin = maad.features.frequency_entropy(Sxx_power)
        result['Ht_per_bin'] = Ht_per_bin

    if 'EVNspCount_per_bin' in indices:
        _, _, EVNspCount_per_bin, _ = maad.features.spectral_events(Sxx_dB_noNoise,
                                                                    dt=tn[1] - tn[0], dB_threshold=6, rejectDuration=0.1, display=True, extent=ext)
        result['EVNspCount_per_bin'] = EVNspCount_per_bin

    return result

def create_fcs(indices, files, unit_fcs):
    df = create_dataframe_from_wav_files(files)

    index_data = []

    for file in files:
        indices_result = calculate_indices(file, indices)
        index_data.append(indices_result)

    print(index_data)

    df_indices = pd.DataFrame(index_data)
    df_indices.set_index(df.index, inplace=True)
    df = pd.concat([df, df_indices], axis=1)
    df = df.drop(columns=['file'])
    fcs, triplet = false_Color_Spectro(
        df=df,
        indices=indices,
        reverseLUT=False,
        unit=unit_fcs,
        permut=False,
        display=True,
        figsize=(12, 6),
        savefig='abc'
    )
    plt.close('all')
    print("Minimum Datetime:", df.index.min())
    print("Maximum Datetime:", df.index.max())
    print(df)
    #print(df)
    return fcs