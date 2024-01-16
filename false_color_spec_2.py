import re
import os
import pandas as pd
from maad.util import (
    date_parser, plot_correlation_map,
    plot_features_map, plot_features, false_Color_Spectro
    )
from maad import sound, features
import matplotlib.pyplot as plt

df_indices_per_bin = pd.read_csv('example.csv')

directory_path = '/Users/ridasaghir/Desktop/24_hour'
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

def create_fcs(indices, files, unit_fcs):
    df = create_dataframe_from_wav_files(files)
    df_indices = pd.DataFrame()
    df_indices_per_bin = pd.DataFrame()

    for index, row in df.iterrows():

        # get the full filename of the corresponding row
        fullfilename = row['file']
        # Save file basename
        path, filename = os.path.split(fullfilename)
        print('\n**************************************************************')
        print(filename)

        S = -35  # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)
        G = 26 + 16  # Amplification gain (26dB (SM4 preamplifier))

        #### Load the original sound (16bits) and get the sampling frequency fs
        try:
            wave, fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

        except:
            # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
            df.drop(index, inplace=True)
            continue

        # Compute the Power Spectrogram Density (PSD) : Sxx_power
        Sxx_power, tn, fn, ext = sound.spectrogram(
            x=wave,
            fs=fs,
            window='hann',
            nperseg=1024,
            noverlap=1024 // 2,
            verbose=False,
            display=False,
            savefig=None
        )

        df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(
            Sxx_power=Sxx_power,
            tn=tn,
            fn=fn,
            flim_low=[0, 1500],
            flim_mid=[1500, 8000],
            flim_hi=[8000, 20000],
            gain=G,
            sensitivity=S,
            verbose=False,
            R_compatible='soundecology',
            mask_param1=6,
            mask_param2=0.5,
            display=False)

        # First, we create a dataframe from row that contains the date and the
        # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
        # then transposing the DataFrame.
        df_row = pd.DataFrame(row)
        df_row = df_row.T
        df_row.index.name = 'Date'
        df_row = df_row.reset_index()

        # create a row with the different vector indices
        row_vector_indices = pd.concat(
            [df_row, df_spec_ind_per_bin],
            axis=1)

        # add vector indices into the df_indices_per_bin dataframe
        df_indices_per_bin = pd.concat([df_indices_per_bin, row_vector_indices])

    # # Set back Date as index
    df_indices_per_bin = df_indices_per_bin.set_index('Date')

    fcs, triplet = false_Color_Spectro(
        df=df_indices_per_bin,
        indices = indices,
        reverseLUT=False,
        unit=unit_fcs,
        permut=False,
        display=True,
        figsize=(12,6),
        savefig='spec'
        )
    for f in os.listdir('.'):
        if os.path.isfile(f) and f.startswith('spec'):
            return f



