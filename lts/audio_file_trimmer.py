import os
import re
import pandas as pd
import librosa
import soundfile as sf
from utils import load_config
def trimmer(row, resolution):
    """
    Parses the file in the data frame according to timestamp and resolution'

    Args:
        row : A row from the data frame including file name and time stamp.
        resolution: The resolution selected for feature calculation.
    Returns:
        output: Tuple containing (y, sr). y is trimmed
    """
    _, path_data, last_dataset, path_exp, _, _, _, _, _, _ = load_config()
    file_path = os.path.join(path_data, last_dataset, row['File Name'])
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    start_time = pd.to_datetime(row['Date Time']).second
    result = re.search(r'\d+', resolution)
    resolution = int(result.group())
    end_time = start_time + resolution
    start_index = int(start_time * sr)
    end_index = int(end_time * sr)
    trimmed_audio = y[start_index:end_index]
    output = (trimmed_audio, sr)
    return output

