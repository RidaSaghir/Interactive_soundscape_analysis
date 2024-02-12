from datetime import datetime
import re
from maad import sound, util
import matplotlib.pyplot as plt
from pydub import AudioSegment
from maad.rois import find_rois_cwt
from maad.util import plot_spectrogram

class AcousticRegionAnalyzer():
    def __init__(self):
        pass
    def find_acoustic_region(self, timestamp):
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
    def acoustic_regions(self, files):
        regions = []

        for file in files:
            self.s, self.fs = sound.load(file)
            Sxx, tn, fn, ext = sound.spectrogram(self.s, self.fs, nperseg=1024, noverlap=512)
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
                region_found = self.find_acoustic_region(timestamp)
                regions.append(region_found)

        return fig, regions

    def acoustic_reg_roi(self, region, file):
        numeric_value = re.findall(r'\d+', region)
        s, fs = sound.load(file[0])
        fig, ax = plt.subplots(figsize=(10, 5))

        if (numeric_value[0] in ['4', '8', '12', '16', '20']):
            self.df_trill = find_rois_cwt(s, fs, flims=(7906, 11000), tlen=2, th=0, display=True, figsize=(10, 6))
            plt.show()
            a = 'roi.png'
            plt.savefig(a)

        elif (numeric_value[0] in ['3' or '7' or '11' or '15' or '19']):
            self.df_trill = find_rois_cwt(s, fs, flims=(3609, 7906), tlen=2, th=0, display=True, figsize=(10, 6))
            plt.show()
            a = 'roi.png'
            plt.savefig(a)

        elif (numeric_value[0] in ['2' or '6' or '10' or '14' or '18']):
            self.df_trill = find_rois_cwt(s, fs, flims=(988, 3609), tlen=2, th=0, display=True, figsize=(10, 6))
            plt.show()
            a = 'roi.png'
            plt.savefig(a)

        elif (numeric_value[0] in ['1' or '5' or '9' or '13' or '17']):
            # 0.5 because the function expects values greater than 0
            self.df_trill = find_rois_cwt(s, fs, flims=(0.1, 988), tlen=2, th=0, display=True, figsize=(10, 6))
            plt.show()
            a = 'roi.png'
            plt.savefig(a)

        rois_num = []
        for roi in self.df_trill.index.tolist():
            rois_num.append(f"ROI {roi}")

        return a, rois_num

    def roi_play(self, roi_selected):
        numeric_value = re.findall(r'\d+', roi_selected)
        print(self.df_trill)
        print(numeric_value)
        min_f = self.df_trill.loc[int(numeric_value[0]), 'min_f']
        max_f = self.df_trill.loc[int(numeric_value[0]), 'max_f']
        min_t = self.df_trill.loc[int(numeric_value[0]), 'min_t']
        max_t = self.df_trill.loc[int(numeric_value[0]), 'max_t']
        start_idx = int(min_t * self.fs)
        end_idx = int(max_t * self.fs)
        s_filtered = sound.select_bandwidth(self.s, self.fs, fcut=(min_f, max_f), forder=5, fname='butter', ftype='bandpass')
        cropped_segment = AudioSegment(s_filtered[start_idx:end_idx].tobytes(), frame_rate=self.fs, sample_width=2, channels=1)
        name_aud = "audio_cropped.wav"
        cropped_segment.export(name_aud, format="wav")
        return name_aud



