import numpy as np
from gradio_app_local.aci_in_region import compute_ACI
import pandas as pd
import os
import librosa
from scipy import signal
from datetime import datetime
import maad
from maad import features, sound, util


def calculate_values(folder_path, audio_file, hour, minute, timestamp):

    audio_path = os.path.join(folder_path, audio_file)  # Create the full path to the audio file
    y, sr = librosa.load(audio_path, sr=None)
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

