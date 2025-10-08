import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

#############################################
# 1. Integration: Data Loading Functions
#############################################

def load_empatica_file(file_path):
    """
    Load an Empatica E4 file.
    The first row contains the session start time (UTC), the second row the sampling rate.
    The rest of the file contains the recorded data.
    """
    with open(file_path, 'r') as f:
        initial_time = f.readline().strip()        # initial time (UTC)
        sample_rate = f.readline().strip()           # sampling rate in Hz
    # Load the rest of the CSV data after skipping the first two rows.
    df = pd.read_csv(file_path, skiprows=2, header=None)
    return initial_time, float(sample_rate), df

def load_non_eeg_utd_data(file_path):
    """
    Load UT Dallas non-EEG data.
    Converts the 'Timestamp' column to datetime and assigns experimental stage labels.
    """
    df = pd.read_csv(file_path)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    experiment_start = df['Timestamp'].iloc[0]

    stages = [
        ("First Relaxation", timedelta(minutes=5)),
        ("Physical Stress", timedelta(minutes=5)),
        ("Second Relaxation", timedelta(minutes=5)),
        ("Mini-Emotional Stress", timedelta(seconds=40)),
        ("Cognitive Stress", timedelta(minutes=5)),
        ("Third Relaxation", timedelta(minutes=5)),
        ("Emotional Stress", timedelta(minutes=6)),
        ("Fourth Relaxation", timedelta(minutes=5))
    ]

    stage_boundaries = []
    current_time = experiment_start
    for stage, duration in stages:
        stage_boundaries.append((current_time, current_time + duration))
        current_time += duration

    def assign_stage(ts):
        for (start, end), (label, _) in zip(stage_boundaries, stages):
            if start <= ts < end:
                return label
        return "Unknown"

    df['Stage'] = df['Timestamp'].apply(assign_stage)
    return df


def load_all_processed_datasets(test_type, root_path):
    """
    Load all processed dataset CSV files for a given test type from a folder by scanning the directory.
    
    Expected filename format (inspired by the MATLAB file):
      ProcessedData_SubjectXX_{test_type}.csv
    The function splits each filename by '_' and extracts the subject number from the token "SubjectXX".
    Files whose name contains the designated test type are loaded and stored in a dictionary keyed by the subject number.
    
    Parameters:
      test_type (str): The test type string contained in the filename (for example, "PEEP" or "v1").
      root_path (str): The directory where the processed CSV files are located.
      
    Returns:
      processed_data (dict): A dictionary mapping subject numbers (as strings) to their corresponding DataFrame.
    """
    processed_data = {}
    
    # Iterate through all files in the directory.
    for file in os.listdir(root_path):
        if not file.lower().endswith('.csv'):
            continue  # Skip non-CSV files.
        
        # Split the file name using the underscore delimiter.
        tokens = file.split('_')
        
        # We expect at least three tokens: "ProcessedData", "SubjectXX", and "{test_type}.csv"
        if len(tokens) < 3:
            continue
        
        # Check that the prefix is right.
        if tokens[0] != "ProcessedData":
            continue  # Skip files not following the naming convention.
        
        # Extract the subject token (e.g., "Subject20" or "Subject01")
        subject_token = tokens[1]
        if not subject_token.startswith("Subject"):
            continue
        
        # Extract the subject number by stripping the "Subject" part.
        subject_number = subject_token[len("Subject"):]
        
        # Check if the next token contains the test_type.
        # Depending on your exact naming scheme, tokens[2] might include the file extension.
        if test_type not in tokens[2]:
            continue
        
        file_path = os.path.join(root_path, file)
        try:
            df = pd.read_csv(file_path)
            processed_data[subject_number] = df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return processed_data



def load_subject_info(info_path):
    """
    Load subject-info.csv containing demographic and clinical details.
    """
    try:
        df = pd.read_csv(info_path)
        return df
    except Exception as e:
        print(f"Error loading subject-info.csv: {e}")
        return None

#############################################
# 2. Additional Wearable Signal Functions (from the ipynb)
#############################################

def moving_average(acc_data):
    """
    Calculate a moving average for accelerometer (ACC) data.
    Processes the data in 32-sample chunks, computing a max change per sample
    and then applying exponential smoothing.
    """
    avg = 0
    prevX, prevY, prevZ = 0, 0, 0
    results = []
    for i in range(0, len(acc_data), 32):
        sum_ = 0
        buffX = acc_data[i:i+32, 0]
        buffY = acc_data[i:i+32, 1]
        buffZ = acc_data[i:i+32, 2]
        for j in range(len(buffX)):
            sum_ += max(
                abs(buffX[j] - prevX),
                abs(buffY[j] - prevY),
                abs(buffZ[j] - prevZ)
            )
            prevX, prevY, prevZ = buffX[j], buffY[j], buffZ[j]
        avg = avg * 0.9 + (sum_ / 32) * 0.1
        results.append(avg)
    return results

def graph_multiple(signals, timeline, subject_signals, state):
    """
    Plot multiple signals for a given subject from one recording session.
    For ACC data, apply a moving average before plotting.
    Draw vertical markers based on the 'tags' signal.
    """
    plt.figure(figsize=(25, 15))
    
    keys = list(signals[subject_signals].keys())
    if "tags" in keys:
        keys.remove("tags")
    
    i = 1
    for key in keys:
        plt.subplot(len(keys), 1, i)
        if i == 1:
            plt.title(f"{subject_signals}  -  {state}")
        if key == 'ACC':
            acc = moving_average(signals[subject_signals][key])
            plt.plot(acc, label=key)
        else:
            plt.plot(timeline[subject_signals][key], signals[subject_signals][key], label=key)
            
        if "tags" in signals[subject_signals]:
            for tag in signals[subject_signals]["tags"][1:]:
                plt.axvline(x=tag, color='r', linestyle='-')
            if state == 'STRESS' and signals[subject_signals]["tags"]:
                if 'S' in subject_signals:  # first version
                    plt.axvspan(signals[subject_signals]["tags"][3], signals[subject_signals]["tags"][4], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][5], signals[subject_signals]["tags"][6], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][7], signals[subject_signals]["tags"][8], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][9], signals[subject_signals]["tags"][10], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][11], signals[subject_signals]["tags"][12], color='red', alpha=0.2)
                else:  # second version
                    plt.axvspan(signals[subject_signals]["tags"][2], signals[subject_signals]["tags"][3], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][4], signals[subject_signals]["tags"][5], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][6], signals[subject_signals]["tags"][7], color='red', alpha=0.2)
                    plt.axvspan(signals[subject_signals]["tags"][8], signals[subject_signals]["tags"][9], color='red', alpha=0.2)
        plt.legend()
        plt.grid()
        i += 1
    plt.show()

def create_df_array(dataframe):
    """
    Create a one-dimensional numpy array from a DataFrame's values.
    """
    matrix_df = dataframe.values
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()
    return array_df

def time_abs_(UTC_array):
    """
    Convert a list of UTC timestamp strings into relative seconds (0 at the start).
    """
    new_array = []
    for utc in UTC_array:
        time_diff = (
            datetime.datetime.strptime(utc, '%Y-%m-%d %H:%M:%S') -
            datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')
        ).total_seconds()
        new_array.append(int(time_diff))
    return new_array

def read_signals(main_folder):
    """
    Reads wearable sensor signals from a main folder containing subject subfolders.
    For each subject, expects CSV files: 'EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', and 'ACC.csv'.
    
    Returns three dictionaries:
      - signal_dict: raw signals for each subject.
      - time_dict:   time arrays (in seconds) for each signal.
      - fs_dict:     sampling frequencies for each signal.
    """
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    # List subject folders under the main folder.
    subfolders = next(os.walk(main_folder))[1]

    utc_start_dict = {}
    # For each subject, record the header (UTC start) from EDA.csv.
    for folder_name in subfolders:
        csv_path = os.path.join(main_folder, folder_name, "EDA.csv")
        df = pd.read_csv(csv_path)
        utc_start_dict[folder_name] = df.columns.tolist()

    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        files = os.listdir(folder_path)

        signals = {}
        time_line = {}
        fs_signal = {}
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', 'ACC.csv']

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        # Insert the UTC timestamp extracted earlier.
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array = []
                else:
                    df = pd.read_csv(file_path)
                    fs = int(df.loc[0, df.columns[0]])
                    df.drop([0], axis=0, inplace=True)
                    signal_array = df.values
                    time_array = np.linspace(0, len(signal_array) / fs, num=len(signal_array))
                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                if file_name != 'tags.csv':
                    time_line[signal_name] = time_array
                    fs_signal[signal_name] = fs
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal

    return signal_dict, time_dict, fs_dict

#############################################
# 3. AI Recreation: Data Processing & Analysis Functions
#############################################

def extract_features_wearable(wearable_data):
    """
    Example feature extraction from wearable signals.
    For demonstration, we simply count the number of events in the tags signal from the STRESS activity.
    Extend this function to compute HRV, EDA responses, or other features.
    """
    features = {}
    try:
        stress_activity = wearable_data['activities'].get("STRESS", {})
        stress_features = {}
        for participant, signals in stress_activity.items():
            if 'tags' in signals:
                tags_df = signals['tags']['data']
                stress_features[participant] = len(tags_df)
        features['wearable_events_count'] = stress_features
    except Exception as e:
        print("Error extracting wearable features:", e)
    return features

def process_integrated_data(data_bundle):
    """
    Process the integrated data to extract features for further AI analysis.
    """
    processed_features = {}
    # Extract wearable features.
    wearable = data_bundle.get("wearable", {})
    processed_features["wearable_features"] = extract_features_wearable(wearable)
    
    # Example: For BIDMC data, record the number of sample points.
    bidmc = data_bundle.get("bidmc", {})
    if bidmc.get("signals") is not None:
        processed_features["bidmc_length"] = bidmc["signals"].shape[0]
    
    # Example: From UT Dallas non-EEG, count the number of samples per experimental stage.
    utd = data_bundle.get("utd_non_eeg")
    if utd is not None:
        stage_counts = utd.groupby("Stage").size().to_dict()
        processed_features["utd_stage_counts"] = stage_counts

    return processed_features

#############################################
# 4. Integration Function: Combine All Datasets
#############################################

def integrate_data():
    """
    Combine different datasets into a single data bundle.
    Uses:
      - BIDMC data (processed datasets)
      - Dexcom-Empatica data (if available)
      - UT Dallas non-EEG data
      - Subject information
      - Wearable stress/exercise signals from "stress-and-exercise-sessions"
    """
    # Define your dataset paths.
    bidmc_root = "bidmc-ppg-and-resp/bidmc_csv"
    # dexcom_empatica_root = "glycemic-variability"  # if applicable
    utd_non_eeg_file = os.path.join("non-eeg", "subjectinfo.csv")
    subject_info_file = os.path.join("real-world-settings", "subject-info.csv")
    processed_root = "resp-and-HR-monitoring-aeration/Processed_Dataset"
    # Path for wearable signals from exercise sessions.
    wearable_root = "stress-and-exercise-sessions"
    
    combined_data = {}

    processed_datasets = load_all_processed_datasets("PEEP", processed_root)
    
      # adjust arguments as needed
    combined_data["utd_non_eeg"] = load_non_eeg_utd_data(utd_non_eeg_file)
    combined_data["subject_info"] = load_subject_info(subject_info_file)
    
    # Load wearable data using the new ipynb function.
    # The wearable_root folder contains subfolders for states (AEROBIC, ANAEROBIC, STRESS).
    states = os.listdir(wearable_root)
    wearable_data = {}
    wearable_time = {}
    wearable_fs = {}
    
    for state in states:
        folder_path = os.path.join(wearable_root, state)
        sig_data, t_data, fs_data = read_signals(folder_path)
        wearable_data[state] = sig_data
        wearable_time[state] = t_data
        wearable_fs[state] = fs_data
    
    combined_data["wearable"] = {
        "signals": wearable_data,
        "time": wearable_time,
        "fs": wearable_fs
    }
    
    return combined_data

#############################################
# Main: Run Integration & (Example) Visualization
#############################################

if __name__ == '__main__':
    # Combine datasets from various sources.
    data_bundle = integrate_data()
    features = process_integrated_data(data_bundle)
    print("Extracted Features:")
    print(features)

    # Example: Visualize wearable signals for the STRESS state.
    wearable_signals = data_bundle["wearable"]["signals"].get("STRESS", {})
    wearable_time = data_bundle["wearable"]["time"].get("STRESS", {})
    if wearable_signals:
        for participant in wearable_signals.keys():
            graph_multiple(wearable_signals, wearable_time, participant, "STRESS")