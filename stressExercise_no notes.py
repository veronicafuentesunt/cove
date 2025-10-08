import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def moving_average(acc_data):
    """
    Computes a moving average over blocks of 32 samples.
    Assumes acc_data is a 2D array with at least 3 columns.
    Returns an array of averaged values.
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
    return np.array(results)

def graph_by_session(subject, state, signal_data, time_data):
    """
    For a given subject and state it creates a figure where each available
    signal channel gets its own subplot.
    
    The plots use a state-specific color:
         STRESS: red, solid,
         AEROBIC: blue, dashed,
         ANEROBIC: green, dotted.
    
    If the signal channel is 'ACC', the moving_average is applied and the timeline
    is downsampled by averaging every 32 samples.
    """
    if subject not in signal_data.get(state, {}):
        print(f"Subject {subject} not available in state {state}")
        return

    available_keys = list(signal_data[state][subject].keys())
    if "tags" in available_keys:
        available_keys.remove("tags")
    n_signals = len(available_keys)

    if state == "STRESS":
        color = "red"
        linestyle = "-"
    elif state == "AEROBIC":
        color = "blue"
        linestyle = "--"
    elif state == "ANEROBIC":
        color = "green"
        linestyle = ":"
    else:
        color = "black"
        linestyle = "-"

    plt.figure(figsize=(25, 5 * n_signals))
    for i, key in enumerate(available_keys, start=1):
        plt.subplot(n_signals, 1, i)
        plt.title(f"{subject} - {state} - Signal: {key}")

        timeline = time_data[state][subject][key]
        data = signal_data[state][subject][key]
        
        # If the signal is ACC, downsample both the data and timeline.
        if key == 'ACC':
            original_timeline = timeline
            new_timeline = []
            # For each block of 32 samples, average the corresponding timeline values.
            for j in range(0, len(original_timeline), 32):
                new_timeline.append(np.mean(original_timeline[j:j+32]))
            timeline = np.array(new_timeline)
            data = moving_average(data)

        plt.plot(timeline, data, label=f"{state}", color=color, linestyle=linestyle, linewidth=2)

        # Plot vertical tag markers if available.
        if "tags" in signal_data[state][subject]:
            for tag in signal_data[state][subject]["tags"][1:]:
                plt.axvline(x=tag, color=color, linestyle=linestyle, alpha=0.5)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
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

def read_signals(main_folder):
    """
    Reads CSV files from a given folder (each representing one session type)
    and returns three dictionaries:
      - signal_dict: signals per subject,
      - time_dict: time arrays per file,
      - fs_dict: sampling frequencies per file.
    """
    signal_dict = {}
    time_dict = {}
    fs_dict = {}
    subfolders = next(os.walk(main_folder))[1]
    utc_start_dict = {}
    
    # Create a map of folder_name -> UTC start info from the EDA.csv header.
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

def analyze_hr_patterns(data_bundle, window=600):
    """
    Analyzes HR patterns for STRESS sessions.
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
            plt.tight_layout()
            plt.draw()  # update the figure
            print("Click on the figure or press any key to continue...")
            plt.waitforbuttonpress()  # wait for interaction
            plt.close()

def analyze_hr_trends_physical(state, data_bundle, window=600):
    """
    Analyzes HR trends for physical sessions (AEROBIC or ANEROBIC).
    For each subject, it finds HR spikes (using a threshold of mean+std),
    then computes the trend (slope) over the window before each spike.
    The function prints the individual slopes and an average per subject.
    """
    if state not in data_bundle["wearable"]["activities"]:
        print(f"No {state} data available in wearable signals.")
        return

    hr_data_dict = data_bundle["wearable"]["activities"][state]
    hr_time_dict = data_bundle["wearable"]["time"][state]

    slopes = {}  # Will store a list of slopes per subject
    for subj, signals in hr_data_dict.items():
        if "HR" not in signals:
            print(f"Subject {subj}: No HR signal available in {state}.")
            continue

        HR = signals["HR"].flatten()
        if subj not in hr_time_dict or "HR" not in hr_time_dict[subj]:
            print(f"Subject {subj}: No HR time data available in {state}.")
            continue

        t = hr_time_dict[subj]["HR"]
        threshold = np.mean(HR) + np.std(HR)
        peaks, properties = find_peaks(HR, height=threshold)
        slopes[subj] = []
        print(f"Subject {subj} in {state}: Detected {len(peaks)} HR spikes (threshold = {threshold:.2f}).")
        
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
            slopes[subj].append(slope)
            print(f"  Spike at {peak_time:.2f}s: Mean: {window_mean:.2f}, Std: {window_std:.2f}, Slope: {slope:.4f}")
    
    print(f"\nAverage HR slopes for {state}:")
    for subj, sub_slopes in slopes.items():
        if sub_slopes:
            avg_slope = np.mean(sub_slopes)
            print(f"  {subj}: {avg_slope:.4f} (from {len(sub_slopes)} spikes)")
        else:
            print(f"  {subj}: No slopes computed.")

if __name__ == '__main__':
    dataset_path = 'stress-and-exercise-sessions'
    wearable_root = os.path.join(dataset_path, "data")
    states = os.listdir(wearable_root)
    signal_data = {}  # Signals per state (folder) and subject.
    time_data = {}
    fs_data = {}
    participants = {}
    
    for state in states:
        folder_path = os.path.join(wearable_root, state)
        participants[state] = os.listdir(folder_path)
        sig, t_d, fs_d = read_signals(folder_path)
        signal_data[state] = sig
        time_data[state] = t_d
        fs_data[state] = fs_d
        
    data_bundle = {
        "wearable": {
            "activities": signal_data,
            "time": time_data,
            "fs": fs_data
        }
    }
    
    # Build the set of all subjects across states.
    all_subjects = set()
    for state in ['STRESS', 'AEROBIC', 'ANEROBIC']:
        all_subjects.update(set(participants.get(state, [])))
    
    # Plot each session type separately per subject.
    for subject in sorted(all_subjects):
        for state in ['STRESS', 'AEROBIC', 'ANEROBIC']:
            graph_by_session(subject, state, signal_data, time_data)
    
    # Analyze HR patterns for mental stress (STRESS).
    analyze_hr_patterns(data_bundle, window=600)
    
    # Analyze HR trends for physical sessions (AEROBIC and ANEROBIC) so you can compare them.
    print("\n--- AEROBIC HR Trends ---")
    analyze_hr_trends_physical("AEROBIC", data_bundle, window=600)
    
    print("\n--- ANEROBIC HR Trends ---")
    analyze_hr_trends_physical("ANEROBIC", data_bundle, window=600)