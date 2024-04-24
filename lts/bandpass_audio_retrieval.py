import os

import librosa
from scipy.signal import butter, lfilter
import soundfile as sf


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_audio(audios, region):
    """
    Bandpass filters the files based on selected region

    Args:
        audios : List of file paths to be filtered.
        region: Selected region like 'Acoustic Region 1'
    Returns:
        filtered_audios: Contains list of tuples of filtered audio and frame rate like (y, sr) for all the files
    """
    region = (region.split()[2])

    if region in ['1', '5', '9', '13', '17']:
        low = 50
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
    for audio in audios:
        if isinstance(audio, tuple):
            y, sr = audio
        else:
            y, sr = librosa.load(audio, sr=None)
        filtered_audio = butter_bandpass_filter(y, low, high, sr, order=5)
        filtered_audios.append((filtered_audio, sr))

    return filtered_audios

