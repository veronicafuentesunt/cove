import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =============================================
# 1. Helper Functions for Data Processing & Visualization
# =============================================

def moving_average(acc_data):
    # Initialization of variables
    avg = 0
    prevX, prevY, prevZ = 0, 0, 0
    results = []
    # Each second (32 samples) the acceleration data is summarized using the following method:
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
        # The output is then filtered:
        avg = avg * 0.9 + (sum_ / 32) * 0.1
        results.append(avg)
    return results

def graph_multiple(signals, timeline, subject_signals, state):
    plt.figure(figsize=(25,15))
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
    matrix_df = dataframe.values
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()  # Convert matrix into a 1D array
    return array_df

def time_abs_(UTC_array):
    new_array = []
    for utc in UTC_array:
        time_diff = (datetime.datetime.strptime(utc, '%Y-%m-%d %H:%M:%S') -
                     datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds()
        new_array.append(int(time_diff))
    return new_array

# =============================================
# 2. Data Loading Functions
# =============================================

def read_signals(main_folder):
    """
    Read wearable sensor CSV files from a given main folder that contains subject subfolders.
    Each subject folder (e.g., "S01", "S02", etc.) should contain:
       'EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', 'ACC.csv'
    Returns:
       signal_dict: Nested dictionary {subject: {signalName: data_array, ...}}
       time_dict: Time vectors for each signal.
       fs_dict: Sampling frequencies for each signal.
    """
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    # Get list of subject folders inside the main folder.
    subfolders = next(os.walk(main_folder))[1]
    
    # Get UTC start from each subject's EDA.csv
    utc_start_dict = {}
    for folder_name in subfolders:
        csv_path = os.path.join(main_folder, folder_name, "EDA.csv")
        df = pd.read_csv(csv_path)
        utc_start_dict[folder_name] = df.columns.tolist()

    # Iterate over subject folders.
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
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array = []
                else:
                    df = pd.read_csv(file_path)
                    fs = int(df.loc[0, df.columns[0]])
                    df.drop([0], axis=0, inplace=True)
                    signal_array = df.values
                    time_array = np.linspace(0, len(signal_array)/fs, num=len(signal_array))
                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array
                fs_signal[signal_name] = fs
        
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal
    
    return signal_dict, time_dict, fs_dict

# =============================================
# 3. HR Pattern Analysis Function (10 minutes before a spike)
# =============================================

def analyze_hr_patterns(data_bundle, window=600):
    """
    Analyze HR patterns for subjects in the STRESS state to determine if there is
    a detectable trend in the 10 minutes (600 seconds) preceding an HR spike.
    
    For each subject in the STRESS state:
      - Extract the HR signal and its time vector.
      - Detect HR spikes using a threshold (mean + std of HR).
      - For each spike, extract the 10-minute window immediately preceding the spike.
      - Compute summary statistics (mean, std, and linear trend slope) for that window.
      - Plot the HR data for that window along with an indication of the spike time.
      
    Parameters:
      data_bundle (dict): Integrated data bundle with wearable signals.
      window (float): Time (in seconds) preceding the spike to analyze (default: 600 seconds).
    """
    if "STRESS" not in data_bundle["wearable"]["activities"]:
        print("No STRESS state data available in wearable signals.")
        return

    hr_data_dict = data_bundle["wearable"]["activities"]["STRESS"]
    hr_time_dict = data_bundle["wearable"]["time"]["STRESS"]

    for subj, signals in hr_data_dict.items():
        if "HR" not in signals:
            print(f"Subject {subj}: No HR signal available.")
            continue

        HR = signals["HR"].flatten()  # Ensure HR is a 1D array
        if subj not in hr_time_dict or "HR" not in hr_time_dict[subj]:
            print(f"Subject {subj}: No HR time data available.")
            continue
        
        t = hr_time_dict[subj]["HR"]
        
        # Define threshold: mean + std of HR
        threshold = np.mean(HR) + np.std(HR)
        peaks, properties = find_peaks(HR, height=threshold)
        print(f"Subject {subj}: Detected {len(peaks)} HR spikes (threshold = {threshold:.2f}).")
        
        for peak in peaks:
            peak_time = t[peak]
            indices = np.where((t >= peak_time - window) & (t <= peak_time))[0]
            if indices.size == 0:
                continue
            hr_window = HR[indices]
            t_window = t[indices]
            
            window_mean = np.mean(hr_window)
            window_std = np.std(hr_window)
            if len(t_window) >= 2:
                slope, intercept = np.polyfit(t_window, hr_window, 1)
            else:
                slope = 0
            
            print(f"Subject {subj} Spike at {peak_time:.2f}s:")
            print(f"  HR Window Mean: {window_mean:.2f}, Std: {window_std:.2f}, Trend Slope: {slope:.4f}")
            
            plt.figure(figsize=(10,4))
            plt.plot(t_window, hr_window, marker='o', label='HR window')
            plt.axvline(t_window[-1], color='r', linestyle='--', label='Spike Time')
            plt.title(f"Subject {subj}: HR pattern 10 minutes before spike at {peak_time:.2f}s")
            plt.xlabel("Time (s)")
            plt.ylabel("HR")
            plt.legend()
            plt.grid(True)
            plt.show()

# =============================================
# 4. Main Integration and Execution
# =============================================

if __name__ == '__main__':
    # Assume your dataset folder is structured as:
    # stress-and-exercise-sessions/
    #   data/
    #      AEROBIC/        <-- Contains subject folders (e.g., F01, F02, S18)
    #      ANAEROBIC/
    #      STRESS/
    
    dataset_path = 'stress-and-exercise-sessions'
    # Set wearable_root to the folder that contains the state folders (e.g., "data")
    wearable_root = os.path.join(dataset_path, "data")
    
    # List state folders (e.g., AEROBIC, ANAEROBIC, STRESS)
    states = os.listdir(wearable_root)
    
    signal_data = {}
    time_data = {}
    fs_data = {}
    participants = {}
    
    # For each state folder, read the signals from its subject subfolders
    for state in states:
        folder_path = os.path.join(wearable_root, state)
        participants[state] = os.listdir(folder_path)
        sig, t_d, fs_d = read_signals(folder_path)
        signal_data[state] = sig
        time_data[state] = t_d
        fs_data[state] = fs_d
    
    # Build the data bundle with wearable information
    data_bundle = {
        "wearable": {
            "activities": signal_data,
            "time": time_data,
            "fs": fs_data
        }
    }
    
    # Optional: Visualize wearable signals for the STRESS state
    if 'STRESS' in participants:
        for person in participants['STRESS']:
            graph_multiple(signal_data['STRESS'], time_data['STRESS'], person, 'STRESS')
    
    # Analyze HR patterns 10 minutes (600 seconds) before HR spikes for subjects in the STRESS state.
    analyze_hr_patterns(data_bundle, window=600)
    
    # Additional plotting (e.g., for self-reported stress levels) could follow here...