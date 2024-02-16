import numpy as np
from datetime import datetime
import maad
from maad import features
aci_values_region = {}
aci_counts_region = {}

def compute_ACI(region, Sxx, freq, fs, time):
    if not isinstance(time, datetime):
        timestamp = datetime.strptime(time, "%H:%M")

    ref_1 = datetime.strptime("5:30", "%H:%M")
    ref_2 = datetime.strptime("9:00", "%H:%M")
    ref_3 = datetime.strptime("17:30", "%H:%M")
    ref_4 = datetime.strptime("21:00", "%H:%M")
    ref_5 = datetime.strptime("23:59", "%H:%M")

    #To initialize
    timestamp = time
    # 0 - 5:29 AM
    if region == 'Region 1' and (timestamp.time() < ref_1.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 1000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 0 - 5:29 AM
    elif region == 'Region 2' and (timestamp.time() < ref_1.time()):
        start_freq = 1000  # Modify this to specify the starting frequency in Hz
        end_freq = 4000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 0 - 5:29 AM
    elif region == 'Region 3' and (timestamp.time() < ref_1.time()):
        start_freq = 4000  # Modify this to specify the starting frequency in Hz
        end_freq = 8000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 0 - 5:29 AM
    elif region == 'Region 4' and (timestamp.time() < ref_1.time()):
        start_freq = 8000  # Modify this to specify the starting frequency in Hz
        end_freq = 11000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 5:30 - 9:00
    elif region == 'Region 5' and (ref_1.time() <= timestamp.time() <= ref_2.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 1000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 5:30 - 9:00
    elif region == 'Region 6' and (ref_1.time() <= timestamp.time() <= ref_2.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 4000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 5:30 - 9:00
    elif region == 'Region 7' and (ref_1.time() <= timestamp.time() <= ref_2.time()):
        start_freq = 4000  # Modify this to specify the starting frequency in Hz
        end_freq = 8000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 5:30 - 9:00
    elif region == 'Region 8' and (ref_1.time() <= timestamp.time() <= ref_2.time()):
        start_freq = 8000  # Modify this to specify the starting frequency in Hz
        end_freq = 11000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 9:01 - 17:29
    elif region == 'Region 9' and (ref_2.time() < timestamp.time() < ref_3.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 1000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 9:01 - 17:29
    elif region == 'Region 10' and (ref_2.time() < timestamp.time() < ref_3.time()):
        start_freq = 1000  # Modify this to specify the starting frequency in Hz
        end_freq = 4000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 9:01 - 17:29
    elif region == 'Region 11' and (ref_2.time() < timestamp.time() < ref_3.time()):
        start_freq = 4000  # Modify this to specify the starting frequency in Hz
        end_freq = 8000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 9:01 - 17:29
    elif region == 'Region 12' and (ref_2.time() < timestamp.time() < ref_3.time()):
        start_freq = 8000  # Modify this to specify the starting frequency in Hz
        end_freq = 11000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 17:30 - 20:59
    elif region == 'Region 13' and (ref_3.time() <= timestamp.time() < ref_4.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 1000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 17:30 - 20:59
    elif region == 'Region 14' and (ref_3.time() <= timestamp.time() < ref_4.time()):
        start_freq = 1000  # Modify this to specify the starting frequency in Hz
        end_freq = 4000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 17:30 - 20:59
    elif region == 'Region 15' and (ref_3.time() <= timestamp.time() < ref_4.time()):
        start_freq = 4000  # Modify this to specify the starting frequency in Hz
        end_freq = 8000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 17:30 - 20:59
    elif region == 'Region 16' and (ref_3.time() <= timestamp.time() < ref_4.time()):
        start_freq = 8000  # Modify this to specify the starting frequency in Hz
        end_freq = 11000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 21:00 - 23:59
    elif region == 'Region 17' and (ref_4.time() <= timestamp.time() <= ref_5.time()):
        start_freq = 0  # Modify this to specify the starting frequency in Hz
        end_freq = 1000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 21:00 - 23:59
    elif region == 'Region 18' and (ref_4.time() <= timestamp.time() <= ref_5.time()):
        start_freq = 1000  # Modify this to specify the starting frequency in Hz
        end_freq = 4000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 21:00 - 23:59
    elif region == 'Region 19' and (ref_4.time() <= timestamp.time() <= ref_5.time()):
        start_freq = 4000  # Modify this to specify the starting frequency in Hz
        end_freq = 8000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    # 21:00 - 23:59
    elif region == 'Region 20' and (ref_4.time() <= timestamp.time() <= ref_5.time()):
        start_freq = 8000  # Modify this to specify the starting frequency in Hz
        end_freq = 11000  # Modify this to specify the ending frequency in Hz
        start_bin = int(start_freq * len(freq) / fs)
        end_bin = int(end_freq * len(freq) / fs)
        Sxx_region = Sxx[start_bin:end_bin + 1, :]
        # Acoustic complexity index
        _, aci_per_bin, aci_main = maad.features.acoustic_complexity_index(Sxx_region)
        min_ACI = min(aci_per_bin)
        max_ACI = max(aci_per_bin)
        # Normalize ACI values to [0, 1]
        normalized_ACI = [(aci - min_ACI) / (max_ACI - min_ACI) for aci in aci_per_bin]
        aci = np.average(normalized_ACI)
        return aci

    else:
        aci = 0
        return aci
