import os

import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import soundfile as sf


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def filtered_audio(files, region):
    region_dir = f'{region}'
    output_dir = os.path.join(os.path.dirname(files[0]), region_dir)

    region = (region.split()[2])

    if region in ['1', '5', '9', '13', '17']:
        low = 0
        high = 988
    elif region in ['2', '6', '10', '14', '18']:
        low = 988
        high = 3609
    elif region in ['3', '7', '11', '15', '19']:
        low = 3609
        high = 7906
    elif region in ['4', '8', '12', '16', '20']:
        low = 7906
        high = 11000

    filtered_audios = []
    for file in files:
        y, sr = librosa.load(file, sr=None)
        filtered_audio = butter_bandpass_filter(y, low, high, sr, order=5)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, os.path.basename(file))
        sf.write(output_file, filtered_audio, sr)
        filtered_audios.append(output_file)

    return filtered_audios

