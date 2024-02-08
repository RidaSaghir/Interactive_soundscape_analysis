from datetime import datetime
import re
from maad import sound, util
import matplotlib.pyplot as plt

from maad.rois import find_rois_cwt
from maad.util import plot_spectrogram



def find_acoustic_region(timestamp):
    ref_1 = datetime.strptime("5:30", "%H:%M")
    ref_2 = datetime.strptime("9:00", "%H:%M")
    ref_3 = datetime.strptime("17:30", "%H:%M")
    ref_4 = datetime.strptime("21:00", "%H:%M")
    ref_5 = datetime.strptime("23:59", "%H:%M")

    region_found = 0
    if (ref_4.time() <= timestamp.time() <= ref_5.time()):
        region_found = ['17', '18', '19', '20']

    elif (timestamp.time() < ref_1.time()):
        region_found = ['1', '2', '3', '4']

    elif (ref_1.time() <= timestamp.time() <= ref_2.time()):
        region_found = ['5', '6', '7', '8']

    elif (ref_2.time() < timestamp.time() < ref_3.time()):
        region_found = ['9', '10', '11', '12']

    elif (ref_3.time() <= timestamp.time() < ref_4.time()):
        region_found = ['13', '14', '15', '16']

    return region_found
def acoustic_regions(files):
    regions = []

    for file in files:
        s, fs = sound.load(file)
        Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)
        ax = util.plot_spectrogram(Sxx, ext, db_range=50, gain=30, figsize=(4,10))
        fig = 'spect.png'
        plt.savefig(fig)

        date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'
        match = re.search(date_time_pattern, file)
        if match:
            year, month, day, hour, minute, second = match.groups()
            year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))
            # Datetime object
            timestamp = datetime(year, month, day, hour, minute, second)
            print(timestamp)
            region_found = find_acoustic_region(timestamp)
            regions.append(region_found)

    return fig, regions
