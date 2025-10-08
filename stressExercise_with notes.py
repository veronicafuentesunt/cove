import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------------------------
# Function: moving_average
# This function computes an exponential moving average of the accelerometer data.
# It processes data in chunks of 32 samples and smooths the signal by combining 
# the previous moving average with the current chunk's average change.
# ------------------------------------------------------------------------------
def moving_average(acc_data):
    avg = 0  # Initialize the smoothed average value
    prevX, prevY, prevZ = 0, 0, 0  # Previous sample's X, Y, Z values - start at 0
    results = []  # This will hold the smoothed values for each chunk
    
    # Process the accelerometer data in chunks of 32 samples
    for i in range(0, len(acc_data), 32):
        sum_ = 0  # Reset sum for the current chunk
        # Extract the next 32 samples for each axis
        buffX = acc_data[i:i+32, 0]
        buffY = acc_data[i:i+32, 1]
        buffZ = acc_data[i:i+32, 2]
        
        # Compute the sum of maximum absolute differences for this chunk
        for j in range(len(buffX)):
            diff = max(
                abs(buffX[j] - prevX),
                abs(buffY[j] - prevY),
                abs(buffZ[j] - prevZ)
            )
            sum_ += diff
            # Update the previous values to the current sample
            prevX, prevY, prevZ = buffX[j], buffY[j], buffZ[j]
        
        # Update the moving average with exponential smoothing (90% old, 10% new)
        avg = avg * 0.9 + (sum_ / 32) * 0.1
        results.append(avg)
    return results

# ------------------------------------------------------------------------------
# Function: graph_multiple
# This function generates multiple subplots of various physiological signals 
# for a specific subject. It uses timelines and overlays 'tags' (markers) to 
# highlight events like stress intervals.
# ------------------------------------------------------------------------------
def graph_multiple(signals, timeline, subject_signals, state):
    plt.figure(figsize=(25,15))  # Create a large figure for clear visualization
    # Get the list of signal keys (e.g., 'HR', 'BVP', etc.) for the subject
    keys = list(signals[subject_signals].keys())
    if "tags" in keys:
        keys.remove("tags")  # Remove 'tags' from the list since they are used for marking events
    i = 1  # Counter for subplots
    for key in keys:
        # Create a separate subplot for each signal type
        plt.subplot(len(keys), 1, i)
        if i == 1:
            plt.title(f"{subject_signals}  -  {state}")  # Title on the first plot
        
        # If the signal is accelerometer data, apply moving average smoothing before plotting
        if key == 'ACC':
            acc = moving_average(signals[subject_signals][key])
            plt.plot(acc, label=key)
        else:
            # For other signals, plot using the provided timeline
            plt.plot(timeline[subject_signals][key], signals[subject_signals][key], label=key)
        
        # Draw vertical lines for each tag (starting from second element) to mark events
        for tag in signals[subject_signals]["tags"][1:]:
            plt.axvline(x=tag, color='r', linestyle='-')
        
        # If in STRESS state, add shaded spans to indicate specific stress intervals
        if state == 'STRESS' and signals[subject_signals]["tags"]:
            if 'S' in subject_signals:  # For one version of subject identifiers
                plt.axvspan(signals[subject_signals]["tags"][3], signals[subject_signals]["tags"][4], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][5], signals[subject_signals]["tags"][6], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][7], signals[subject_signals]["tags"][8], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][9], signals[subject_signals]["tags"][10], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][11], signals[subject_signals]["tags"][12], color='red', alpha=0.2)
            else:  # For an alternate version of subject identifiers
                plt.axvspan(signals[subject_signals]["tags"][2], signals[subject_signals]["tags"][3], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][4], signals[subject_signals]["tags"][5], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][6], signals[subject_signals]["tags"][7], color='red', alpha=0.2)
                plt.axvspan(signals[subject_signals]["tags"][8], signals[subject_signals]["tags"][9], color='red', alpha=0.2)
        plt.legend()
        plt.grid()
        i += 1  # Move to the next subplot
    plt.show()  # Display all the subplots

# ------------------------------------------------------------------------------
# Function: create_df_array
# This utility function converts a DataFrame's values into a 1D NumPy array.
# ------------------------------------------------------------------------------
def create_df_array(dataframe):
    matrix_df = dataframe.values  # Get all values as a NumPy array
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()  # Flatten the 2D array into a 1D array
    return array_df

# ------------------------------------------------------------------------------
# Function: time_abs_
# Convert an array of UTC timestamp strings into an array of seconds passed
# relative to the first timestamp.
# ------------------------------------------------------------------------------
def time_abs_(UTC_array):
    new_array = []
    for utc in UTC_array:
        # Parse the timestamp string into a datetime object and compute difference from the first timestamp
        time_diff = (datetime.datetime.strptime(utc, '%Y-%m-%d %H:%M:%S') -
                     datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds()
        new_array.append(int(time_diff))  # Append the difference in seconds as an integer
    return new_array

# ------------------------------------------------------------------------------
# Function: read_signals
# Reads signals from subfolders in the main folder, converts each relevant CSV file 
# into NumPy arrays, and creates corresponding time arrays based on the sampling frequency.
# ------------------------------------------------------------------------------
def read_signals(main_folder):
    signal_dict = {}  # Dictionary for signal data per subject
    time_dict = {}    # Dictionary for time arrays per subject
    fs_dict = {}      # Dictionary for sampling frequencies per subject
    # Get list of subfolders (each subfolder corresponds to a subject/session)
    subfolders = next(os.walk(main_folder))[1]
    utc_start_dict = {}
    
    # First, retrieve UTC start information from the "EDA.csv" headers
    for folder_name in subfolders:
        csv_path = os.path.join(main_folder, folder_name, "EDA.csv")
        df = pd.read_csv(csv_path)
        utc_start_dict[folder_name] = df.columns.tolist()  # Save header as UTC info
    
    # Process each subject's folder
    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        files = os.listdir(folder_path)
        signals = {}   # Holds signal arrays for current subject
        time_line = {} # Holds corresponding time arrays
        fs_signal = {} # Holds sampling frequencies
        
        # Define the files we want to process
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', 'ACC.csv']
        
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            # Process only desired files that end with '.csv'
            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        # Read tags without header and flatten to 1D array
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        # Insert UTC start info at the beginning to form a full time vector
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array = []  # Handle empty data gracefully
                else:
                    # For all other files, read the CSV into a DataFrame
                    df = pd.read_csv(file_path)
                    # The first row contains the sampling frequency (fs)
                    fs = int(df.loc[0, df.columns[0]])
                    df.drop([0], axis=0, inplace=True)  # Drop the row that contains fs
                    signal_array = df.values  # Convert remaining data to a NumPy array
                    # Create a time array using linear spacing from 0 to duration (samples / fs)
                    time_array = np.linspace(0, len(signal_array) / fs, num=len(signal_array))
                # Use file name (without extension) as the key for the signals dictionary
                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array
                fs_signal[signal_name] = fs
        
        # Save the processed data for the current subject in the corresponding dictionaries
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal
    
    return signal_dict, time_dict, fs_dict

# ------------------------------------------------------------------------------
# Function: analyze_hr_patterns
# Analyzes heart rate (HR) patterns for subjects in the STRESS session.
# It detects HR spikes using a dynamic threshold and examines the HR behavior
# in the 10-minute (600-second) period before each spike.
# ------------------------------------------------------------------------------
def analyze_hr_patterns(data_bundle, window=600):
    # Check if there is STRESS state data in the provided data bundle.
    if "STRESS" not in data_bundle["wearable"]["activities"]:
        print("No STRESS state data available in wearable signals.")
        return

    hr_data_dict = data_bundle["wearable"]["activities"]["STRESS"]
    hr_time_dict = data_bundle["wearable"]["time"]["STRESS"]

    # Loop over each subject in the STRESS state
    for subj, signals in hr_data_dict.items():
        if "HR" not in signals:
            print(f"Subject {subj}: No HR signal available.")
            continue

        # Flatten HR data into a 1D array for analysis
        HR = signals["HR"].flatten()
        if subj not in hr_time_dict or "HR" not in hr_time_dict[subj]:
            print(f"Subject {subj}: No HR time data available.")
            continue
        
        # Retrieve the time vector for HR data
        t = hr_time_dict[subj]["HR"]
        # Set a dynamic threshold: mean HR plus one standard deviation
        threshold = np.mean(HR) + np.std(HR)
        # Detect peaks in HR that exceed the threshold
        peaks, properties = find_peaks(HR, height=threshold)
        print(f"Subject {subj}: Detected {len(peaks)} HR spikes (threshold = {threshold:.2f}).")
        
        # Analyze each detected spike
        for peak in peaks:
            peak_time = t[peak]
            # Get indices for samples in the 10-minute window (600 seconds) before the spike
            indices = np.where((t >= peak_time - window) & (t <= peak_time))[0]
            if indices.size == 0:
                continue
            hr_window = HR[indices]
            t_window = t[indices]
            
            # Calculate mean and standard deviation in the window
            window_mean = np.mean(hr_window)
            window_std = np.std(hr_window)
            # Compute trend slope (using linear regression) if there are sufficient data points
            if len(t_window) >= 2:
                slope, intercept = np.polyfit(t_window, hr_window, 1)
            else:
                slope = 0
            
            print(f"Subject {subj} Spike at {peak_time:.2f}s:")
            print(f"  HR Window Mean: {window_mean:.2f}, Std: {window_std:.2f}, Trend Slope: {slope:.4f}")
            
            # Plot the HR window with a marker for the spike time
            plt.figure(figsize=(10, 4))
            plt.plot(t_window, hr_window, marker='o', label='HR window')
            plt.axvline(t_window[-1], color='r', linestyle='--', label='Spike Time')
            plt.title(f"Subject {subj}: HR pattern 10 minutes before spike at {peak_time:.2f}s")
            plt.xlabel("Time (s)")
            plt.ylabel("HR")
            plt.legend()
            plt.grid(True)
            plt.show()

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Define the dataset path and construct the root directory for wearable data
    dataset_path = 'stress-and-exercise-sessions'
    wearable_root = os.path.join(dataset_path, "data")
    
    # Get the list of session states (e.g., STRESS, EXERCISE) from the wearable data folder
    states = os.listdir(wearable_root)
    signal_data = {}  # Dictionary to store signal arrays for all states
    time_data = {}    # Dictionary to store corresponding time arrays
    fs_data = {}      # Dictionary to store sampling frequencies for signals
    participants = {} # Dictionary to store subject IDs per state
    
    # Loop through each state folder and read its signals
    for state in states:
        folder_path = os.path.join(wearable_root, state)
        participants[state] = os.listdir(folder_path)  # List of subjects in the state
        sig, t_d, fs_d = read_signals(folder_path)  # Read signals using our utility function
        signal_data[state] = sig
        time_data[state] = t_d
        fs_data[state] = fs_d
    
    # Package the signals, time, and sampling frequencies into a data bundle for easy reference
    data_bundle = {
        "wearable": {
            "activities": signal_data,
            "time": time_data,
            "fs": fs_data
        }
    }
    
    # For subjects in the STRESS state, generate plots of all their signals with event markers
    if 'STRESS' in participants:
        for person in participants['STRESS']:
            graph_multiple(signal_data['STRESS'], time_data['STRESS'], person, 'STRESS')
    
    # Analyze heart rate patterns for the STRESS state over a 600-second (10 minutes) window
    analyze_hr_patterns(data_bundle, window=600)