import os
import re
import csv
from table_calculations import calculate_values
from datetime import datetime

folder_path = ('/Users/ridasaghir/Desktop/git_hf_noronha_full/noronha_full_01')
#folder_path = ('audio_files')
output_csv = 'parsed_info.csv'

def parse_audio_files(folder_path, output_csv):
    # Validate if the provided path is a directory
    if os.path.isdir(folder_path):
        # List files in the directory
        files = os.listdir(folder_path)
        audio_files = sorted([file for file in files if file.endswith('.wav')])
        parsed_info = []

        # Define a regex pattern to match date and time in filenames
        date_time_pattern = r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})'

        for audio_file in audio_files:
            # Extract date and time using regex pattern

            match = re.search(date_time_pattern, audio_file)
            if match:
                year, month, day, hour, minute, second = match.groups()
                timestamp = f'{hour}:{minute}'
                noise_profile, snr, aci, temp_ent, EVNcount, lfc, mfc, hfc, ACTsp_count, eps, eas, ecv, regions = calculate_values(
                    folder_path, audio_file, hour, minute, timestamp)
                # Calculate the week number
                week_number = datetime(int(year), int(month), int(day)).isocalendar()[1]

                # Calculate the week number within the month
                #day_of_month = int(day)
                #week_number = min(5, (day_of_month - 1) // 7 + 1)

                parsed_info.append({
                    'File': audio_file,
                    'Year': year,
                    'Month': month,
                    'Week': week_number,
                    'Day': day,
                    'Timestamp': timestamp,
                    'Hour': hour,
                    'Minute': minute,
                    'BGN': noise_profile,
                    'SNR': snr,
                    'ACI': aci,
                    'ACI in regions': regions,
                    'ENT': temp_ent,
                    'EVN': EVNcount,
                    'LFC': lfc,
                    'MFC': mfc,
                    'HFC': hfc,
                    'ACT': ACTsp_count,
                    'EPS': eps,
                    'EAS': eas,
                    'ECV': ecv
                })
        if parsed_info:
            # Write the parsed information to a CSV file
            with open(output_csv, 'w', newline='') as csvfile:
                fieldnames = ['File', 'Year', 'Month', 'Week', 'Day', 'Timestamp', 'Hour', 'Minute', 'BGN', 'SNR', 'ACI',
                              'ACI in regions', 'ENT', 'EVN', 'LFC',
                              'MFC', 'HFC', 'ACT', 'EPS', 'EAS', 'ECV']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for info in parsed_info:
                    writer.writerow(info)

            return f"Parsed information saved to {output_csv}"
        else:
            return "No audio files found in the selected folder."

    else:
        return "The provided path is not a valid folder."

parse_audio_files(folder_path, output_csv)

