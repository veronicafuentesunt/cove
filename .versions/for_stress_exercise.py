import pandas as pd
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt

def moving_average(acc_data):
    """
    Calculate a moving average for accelerometer (ACC) data.
    The function processes the data in chunks of 32 samples and for each chunk
    computes the maximum change (in absolute terms) among the three axes. An 
    exponential smoothing filter is then applied.
    """
    avg = 0
    prevX, prevY, prevZ = 0, 0, 0
    results = []
    # Process data in blocks of 32 samples
    for i in range(0, len(acc_data), 32):
        sum_ = 0
        buffX = acc_data[i:i+32, 0]
        buffY = acc_data[i:i+32, 1]
        buffZ = acc_data[i:i+32, 2]
        for j in range(len(buffX)):
            sum_ += max(abs(buffX[j] - prevX),
                        abs(buffY[j] - prevY),
                        abs(buffZ[j] - prevZ))
            prevX, prevY, prevZ = buffX[j], buffY[j], buffZ[j]
        # Exponential smoothing filter
        avg = avg * 0.9 + (sum_ / 32) * 0.1
        results.append(avg)
    return results


def graph_multiple(signals, timeline, subject_signals, state):
    """
    Plot multiple signals for a given subject from one recording session.
    
    - For ACC data, the moving average is computed before plotting.
    - Vertical lines (or shaded regions) are drawn using the 'tags' signal.
    
    Parameters:
      signals         : dict of raw signal arrays per subject.
      timeline        : dict of time vectors for each signal.
      subject_signals : key for the subject (e.g., "S01").
      state           : recording state, e.g. "STRESS".
    """
    plt.figure(figsize=(25, 15))
    
    # Get signal keys, excluding the 'tags' signal.
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
            # Plot vertical lines for each tag (ignoring the first element)
            for tag in signals[subject_signals]["tags"][1:]:
                plt.axvline(x=tag, color='r', linestyle='-')
            # For the STRESS state, highlight event regions.
            if state == 'STRESS' and signals[subject_signals]["tags"]:
                # Here we check the subject identifier to decide which version to use.
                if 'S' in subject_signals:  # first version
                    plt.axvspan(signals[subject_signals]["tags"][3], signals[subject_signals]["tags"][4], color='red', alpha=0.2)  # stroop
                    plt.axvspan(signals[subject_signals]["tags"][5], signals[subject_signals]["tags"][6], color='red', alpha=0.2)  # tmct
                    plt.axvspan(signals[subject_signals]["tags"][7], signals[subject_signals]["tags"][8], color='red', alpha=0.2)  # real opinion
                    plt.axvspan(signals[subject_signals]["tags"][9], signals[subject_signals]["tags"][10], color='red', alpha=0.2)  # opposite opinion
                    plt.axvspan(signals[subject_signals]["tags"][11], signals[subject_signals]["tags"][12], color='red', alpha=0.2)  # subtract test
                else:  # second version
                    plt.axvspan(signals[subject_signals]["tags"][2], signals[subject_signals]["tags"][3], color='red', alpha=0.2)  # tmct
                    plt.axvspan(signals[subject_signals]["tags"][4], signals[subject_signals]["tags"][5], color='red', alpha=0.2)  # real opinion
                    plt.axvspan(signals[subject_signals]["tags"][6], signals[subject_signals]["tags"][7], color='red', alpha=0.2)  # opposite opinion
                    plt.axvspan(signals[subject_signals]["tags"][8], signals[subject_signals]["tags"][9], color='red', alpha=0.2)  # subtract test
        plt.legend()
        plt.grid()
        i += 1
    plt.show()


def create_df_array(dataframe):
    """
    Flatten the values of a DataFrame into a one-dimensional numpy array.
    """
    matrix_df = dataframe.values
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()
    return array_df


def time_abs_(UTC_array):
    """
    Convert a sequence of UTC timestamp strings to seconds relative to the start time.
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
    Read the wearable sensor signals from a given folder containing subject subfolders.
    For each subject, it expects CSV files: 'EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv',
    'tags.csv', and 'ACC.csv'.
    
    The function returns three dictionaries:
      - signal_dict : the raw signal arrays for each subject.
      - time_dict   : time vectors (in seconds) for each signal.
      - fs_dict     : sampling frequency for each signal (as read from the first row of the file).
    """
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    # Get a list of subject subfolders under the main folder.
    subfolders = next(os.walk(main_folder))[1]

    utc_start_dict = {}
    # For each subject, read the header (UTC start) from the EDA.csv file.
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
            # Only process files that are CSVs and are in the desired list.
            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        # Prepend the UTC start timestamp from the header of EDA.csv.
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array = []
                else:
                    df = pd.read_csv(file_path)
                    # The first row contains the sampling frequency.
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


if __name__ == '__main__':
    # Define the path to your dataset.
    dataset_path = "stress-and-exercise-sessions"
    
    # Assume the dataset directory contains folders for the different states:
    # AEROBIC, ANAEROBIC, and STRESS.
    states = os.listdir(dataset_path)  # e.g., ['AEROBIC', 'ANAEROBIC', 'STRESS']

    signal_data = {}
    time_data = {}
    fs_dictionary = {}
    participants = {}

    # Process each state folder.
    for state in states:
        folder_path = os.path.join(dataset_path, state)
        participants[state] = os.listdir(folder_path)
        sig_data, t_data, fs_data = read_signals(folder_path)
        signal_data[state] = sig_data
        time_data[state] = t_data
        fs_dictionary[state] = fs_data

    # For example, plot all participants' signal data for the STRESS state.
    for person in participants.get('STRESS', []):
        graph_multiple(signal_data['STRESS'], time_data['STRESS'], person, 'STRESS')

    # Set a plotting style.
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 8))

    # Load self-reported stress levels.
    # Here we assume the file Stress_Level_v1.csv is in the dataset root.
    stress_level_v1_path = os.path.join(dataset_path, "Stress_Level_v1.csv")
    stress_level_v1 = pd.read_csv(stress_level_v1_path, index_col=0)

    # Plot self-reported stress levels for all participants.
    for index, row in stress_level_v1.iterrows():
        plt.plot(stress_level_v1.columns, row, marker='o', label=index)
    plt.title('Self Reported Stress Level - All Participants V1')
    plt.xlabel('Tests')
    plt.ylabel('Stress Level')
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # Calculate and plot the mean self-reported stress level.
    mean_stress_levels = stress_level_v1.mean()
    plt.figure(figsize=(10, 6))
    mean_stress_levels.plot(kind='bar', color='skyblue')
    plt.title('Mean Self Reported Stress Level - All Participants V1')
    plt.xlabel('Tests')
    plt.ylabel('Stress Level')
    plt.show()

    # Plot the stress level for a specific subject.
    subject = 'S01'  # Update this with the desired subject ID.
    subject_data = stress_level_v1.loc[subject]
    plt.figure(figsize=(10, 6))
    plt.plot(subject_data.index, subject_data.values, marker='o', linestyle='-', color='b')
    plt.title(f'Self Reported Stress Level for {subject} - V1')
    plt.xlabel('Tests')
    plt.ylabel('Stress Level')
    plt.grid(True)
    plt.show()