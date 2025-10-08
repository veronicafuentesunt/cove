import os
import numpy as np
import pandas as pd
import datetime
from datetime import timedelta
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt



def load_student_grades(grades_file):
    """
    Loads the student grades from the text file.
    Assumes a CSV-like format with each line: "Student,Grade".
    """
    try:
        grades_df = pd.read_csv(grades_file, sep=',', header=None, names=['Student', 'Grade'])
        print(f"Loaded student grades with shape {grades_df.shape}")
        return grades_df
    except Exception as e:
        print(f"Error loading grades: {e}")
        return None

def load_exam_data(participant_path, exam_type):
    """
    Loads all CSV files and the info.txt from the specified exam folder.
    Returns a dictionary with keys for each modality and info.
    """
    exam_folder = os.path.join(participant_path, exam_type)
    exam_data = {}
    
    # Files we expect in each exam folder
    file_list = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv', 'TEMP.csv']
    
    for file in file_list:
        file_path = os.path.join(exam_folder, file)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                exam_data[file.split('.')[0]] = df
                print(f"Loaded {file} for {exam_type} in {os.path.basename(participant_path)}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"{file} not found in {exam_folder}")
    
    # Load the info file if it exists for metadata purposes
    info_file = os.path.join(exam_folder, 'info.txt')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            exam_data['info'] = f.read()
    
    return exam_data

def parse_unix_timestamp(unix_ts):
    """
    Convert a unix timestamp (assumed in seconds) to a datetime object.
    """
    return pd.to_datetime(unix_ts, unit='s')

def extract_exam_time_window(exam_data, exam_type):
    """
    Determine the exam start and end times based on the HR data timestamps
    and the known exam schedule. All exams start at 9:00 AM.
    """
    hr_df = exam_data.get('HR', None)
    if hr_df is not None and 'timestamp' in hr_df.columns:
        # Using the first timestamp of the HR data as a reference date
        start_ts = hr_df['timestamp'].iloc[0]
        start_time = parse_unix_timestamp(start_ts)
        # Force the time to be 9:00 AM for that day (considering CT/CDT)
        exam_start_time = start_time.replace(hour=9, minute=0, second=0)
        # Determine exam duration by type
        if exam_type.lower() == 'final':
            exam_duration = datetime.timedelta(hours=3)
        else:
            exam_duration = datetime.timedelta(hours=1, minutes=30)
        exam_end_time = exam_start_time + exam_duration
        print(f"Exam {exam_type}: {exam_start_time} to {exam_end_time}")
        return exam_start_time, exam_end_time
    else:
        print("HR data not found or no timestamp column available.")
    return None, None

def filter_data_by_time(df, start_time, end_time, time_col='timestamp'):
    """
    Filter the dataframe rows based on a specified time window.
    Assumes the time column is in unix timestamp format.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[time_col], unit='s')
    return df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]

# Example usage:

# 1. Load student grades
grades_file = "2_AI\real-world-settings\StudentGrades.txt"
student_grades = load_student_grades(grades_file)

# 2. Set the root directory of the unzipped Data folder
data_root = "2_AI\real-world-settings\Data"
participants = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

for participant in participants:
    participant_path = os.path.join(data_root, participant)
    print(f"\nProcessing data for {participant}...")
    for exam in ['Final', 'Midterm 1', 'Midterm 2']:
        exam_data = load_exam_data(participant_path, exam)
        
        # Extract exam-specific time window using HR.csv as reference
        exam_start_time, exam_end_time = extract_exam_time_window(exam_data, exam)
        if exam_start_time and exam_end_time and 'HR' in exam_data:
            hr_filtered = filter_data_by_time(exam_data['HR'], exam_start_time, exam_end_time, time_col='timestamp')
            print(f"Filtered HR data for {exam} (rows): {hr_filtered.shape[0]}")
        
        # Repeat similar filtering for other modalities if needed.
        # You can then further process these signals to extract features for stress prediction.


def load_bidmc_csv(recording_dir, subject_num):
    """
    Loads the BIDMC dataset for a specific subject using the CSV format.
    
    Parameters:
        recording_dir (str): The directory where the BIDMC CSV files are stored.
        subject_num (int): The subject number (##), which appears in the file names.
    
    Returns:
        A dictionary containing the DataFrames for 'signals', 'numerics', and breath annotations.
    """
    subject_str = f"{subject_num:02d}"  # Format subject number as two digits
    # Construct file paths
    signals_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Signals.csv")
    numerics_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Numerics.csv")
    breaths_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Breaths.csv")
    fix_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Fix.txt")
    
    # Load the CSV files
    try:
        signals_df = pd.read_csv(signals_file)
        print(f"Loaded signals from {signals_file} with shape {signals_df.shape}")
    except Exception as e:
        print(f"Error loading signals: {e}")
        signals_df = None

    try:
        numerics_df = pd.read_csv(numerics_file)
        print(f"Loaded numerics from {numerics_file} with shape {numerics_df.shape}")
    except Exception as e:
        print(f"Error loading numerics: {e}")
        numerics_df = None

    try:
        breaths_df = pd.read_csv(breaths_file)
        print(f"Loaded breath annotations from {breaths_file} with shape {breaths_df.shape}")
    except Exception as e:
        print(f"Error loading breaths: {e}")
        breaths_df = None

    # Optional: Load fixed parameters (as text)
    fix_params = None
    if os.path.exists(fix_file):
        try:
            with open(fix_file, "r") as f:
                fix_params = f.read()
            print(f"Loaded fixed parameters from {fix_file}")
        except Exception as e:
            print(f"Error loading fixed parameters: {e}")

    return {
        'signals': signals_df,
        'numerics': numerics_df,
        'breaths': breaths_df,
        'fixed': fix_params
    }

# Example usage:
recording_directory = "2_AI\bidmc-ppg-and-respiration-dataset-1.0.0"
subject_number = 1  # For the first recording
bidmc_data = load_bidmc_csv(recording_directory, subject_number)



def load_empath_data(participant_dir):
    """
    Loads the Empatica Dexcom data for one participant.
    Expects a folder that contains the following CSVs:
    ACC.csv, BVP.csv, Dexcom.csv, EDA.csv, HR.csv, IBI.csv, TEMP.csv, and a Food Log file.
    """
    # Dictionary to store dataframes
    data = {}
    # List of expected feature files
    feature_files = ['ACC.csv', 'BVP.csv', 'Dexcom.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']
    
    for file_name in feature_files:
        file_path = os.path.join(participant_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            # Convert the "Timestamp" column to datetime if it exists
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            data[file_name.split('.')[0]] = df
            print(f"Loaded {file_name} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    
    # Try loading the Food Log file if it exists
    food_log_file = [f for f in os.listdir(participant_dir) if f.startswith("Food_Log")]
    if food_log_file:
        try:
            food_log_path = os.path.join(participant_dir, food_log_file[0])
            food_log_df = pd.read_csv(food_log_path)
            # Convert date columns if necessary
            if 'date' in food_log_df.columns:
                food_log_df['date'] = pd.to_datetime(food_log_df['date']).dt.date
            data['Food_Log'] = food_log_df
            print(f"Loaded {food_log_file[0]} with shape {food_log_df.shape}")
        except Exception as e:
            print(f"Error loading Food Log: {e}")
    
    return data

def load_demographics(root_dir):
    """
    Loads the demographics information provided in 'Demographics.csv' in the root folder.
    """
    demo_path = os.path.join(root_dir, "Demographics.csv")
    try:
        demo_df = pd.read_csv(demo_path)
        print(f"Loaded Demographics with shape {demo_df.shape}")
        return demo_df
    except Exception as e:
        print(f"Error loading Demographics.csv: {e}")
        return None

# Example usage:
root_folder = "2_AI\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"
# Load demographics for later merging
demographics = load_demographics(root_folder)

# Assume participant folders are named '001', '002', ..., '016'
for participant in [f"{i:03d}" for i in range(1, 17)]:
    participant_path = os.path.join(root_folder, participant)
    print(f"\nProcessing participant {participant}...")
    participant_data = load_empath_data(participant_path)
    
    # Example: Access HR data and resample if needed
    if 'HR' in participant_data:
        hr_df = participant_data['HR']
        # For instance, if the HR data is recorded every second, setting the Timestamp as index
        hr_df.set_index('Timestamp', inplace=True)
        # Resample to 1-minute averages as an example
        hr_resampled = hr_df.resample('1T').mean()
        print(f"Resampled HR data shape: {hr_resampled.shape}")
    
    # Further processing can be done here to extract features and align with other modalities.
    # These features could then be fused with other datasets, or used directly in your predictive pipeline.


def load_non_eeg_data(file_path):
    """
    Load a subject's continuous non-EEG recordings from a CSV file.
    
    Expected file columns include:
      - 'Timestamp': The datetime value (after conversion).
      - 'EDA', 'TEMP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'HR', 'SpO2'
    """
    df = pd.read_csv(file_path)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def label_experiment_stages(df, start_time):
    """
    Label the records in df with the corresponding experimental stage.
    
    Parameters:
      df: DataFrame containing a 'Timestamp' column.
      start_time: The start time of the experiment (the first record timestamp).
    
    Returns:
      The DataFrame with an extra 'Stage' column.
    """
    # Define experiment stages and corresponding durations
    stages = [
        ("First Relaxation", timedelta(minutes=5)),
        ("Physical Stress", timedelta(minutes=5)),
        ("Second Relaxation", timedelta(minutes=5)),
        ("Mini-Emotional Stress", timedelta(seconds=40)),
        ("Cognitive Stress", timedelta(minutes=5)),  # 3 min + 2 min Stroop test
        ("Third Relaxation", timedelta(minutes=5)),
        ("Emotional Stress", timedelta(minutes=6)),    # 1 min anticipation + 5 min clip (or adjust as needed)
        ("Fourth Relaxation", timedelta(minutes=5))
    ]
    
    # Create stage boundaries
    stage_boundaries = []
    current_time = start_time
    for stage, duration in stages:
        stage_boundaries.append((current_time, current_time + duration))
        current_time += duration
    
    # Helper: for a given timestamp, find the corresponding stage
    def assign_stage(ts):
        for (stage_start, stage_end), (stage_label, _) in zip(stage_boundaries, stages):
            if stage_start <= ts < stage_end:
                return stage_label
        return "Unknown"
    
    # Apply stage assignment on each record
    df['Stage'] = df['Timestamp'].apply(assign_stage)
    return df

# Example usage:
subject_file = "2_AI\non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"  # CSV containing continuous signals for one subject.
subject_data = load_non_eeg_data(subject_file)

if not subject_data.empty:
    # Assume the first timestamp of the file marks the experiment start
    experiment_start = subject_data['Timestamp'].iloc[0]
    subject_data = label_experiment_stages(subject_data, experiment_start)
    
    # Display the first few rows to verify stage labels.
    print(subject_data.head())


def load_processed_dataset(subject, test_type, root_path):
    """
    Load a processed dataset CSV file for a given subject and test type.
    
    Parameters:
      subject   : str or int (e.g., "01" or 1). Subject number formatted as two digits.
      test_type : str, one of "PEEP", "PEEP_BH", "FEM" indicating the test.
      root_path : str, the path to the 'ProcessedDataset' folder.
      
    Returns:
      DataFrame containing the data for that subject and test.
    """
    subject_str = f"{int(subject):02d}"
    file_name = f"ProcessedData_Subject{subject_str}_{test_type}.csv"
    file_path = os.path.join(root_path, file_name)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Example usage:
processed_root = "2_AI\respiratory-and-heart-rate-monitoring-dataset-from-aeration-study-1.0.0"
subject_data = load_processed_dataset(subject=1, test_type="PEEP", root_path=processed_root)

def load_pq_raw_data(subject, test_type, root_path, raw=False):
    """
    Load a PQ data CSV file.
    
    Parameters:
      subject   : subject number (formatted as two digits).
      test_type : test type string; e.g., "PEEP", "PEEP_BH", or "FEM".
      root_path : path to the 'PQ_rawData' folder.
      raw       : bool, if True, load the raw ADC counts file (with '_raw' suffix).
      
    Returns:
      DataFrame containing the PQ data.
    """
    subject_str = f"{int(subject):02d}"
    suffix = "_raw" if raw else ""
    file_name = f"Subject{subject_str}_{test_type}{suffix}.csv"
    file_path = os.path.join(root_path, file_name)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded file: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Example usage:
pq_raw_root = "2_AI\respiratory-and-heart-rate-monitoring-dataset-from-aeration-study-1.0.0\PQ_rawData"
pq_data = load_pq_raw_data(subject=1, test_type="PEEP", root_path=pq_raw_root, raw=False)


def load_eit_data(subject, test_type, root_path):
    """
    Load an EIT data file in binary format.
    
    Parameters:
      subject   : subject number as two-digit string.
      test_type : test type string, e.g., "PEEP", "PEEP_BH", "FEM".
      root_path : path to the 'EIT_rawData' folder.
      
    Returns:
      A numpy array representing the 32x32 EIT frames over time.
    """
    subject_str = f"{int(subject):02d}"
    file_name = f"S{subject_str}_{test_type}.bin"
    file_path = os.path.join(root_path, file_name)
    
    try:
        # Here we assume the file is saved as binary data in a given format.
        # Adjust the number of frames or data type according to the file specification.
        data = np.fromfile(file_path, dtype=np.float32)
        # For example, if each frame is 32x32 pixels and data is contiguous, reshape accordingly
        num_frames = data.size // (32 * 32)
        data = data.reshape((num_frames, 32, 32))
        print(f"Loaded EIT data from {file_path} with {num_frames} frames")
        return data
    except Exception as e:
        print(f"Error loading EIT data from {file_path}: {e}")
        return None

# Example usage:
eit_root = "2_AI\respiratory-and-heart-rate-monitoring-dataset-from-aeration-study-1.0.0\EIT_rawData"
eit_data = load_eit_data(subject=1, test_type="PEEP", root_path=eit_root)


def load_hrb_data(subject, root_path):
    """
    Load HRB data for a subject from the HRB folder.
    
    Parameters:
      subject   : subject number (e.g., 3).
      root_path : path to the HRB subfolder under HRM_rawData.
      
    Returns:
      DataFrame with HRB data, if available.
    """
    file_name = f"{subject}.txt"
    file_path = os.path.join(root_path, file_name)
    
    try:
        # Assuming comma-separated values with a header row.
        df = pd.read_csv(file_path)
        print(f"Loaded HRB data for subject {subject} from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading HRB data from {file_path}: {e}")
        return None

# Example usage:
hrb_root = "2_AI\respiratory-and-heart-rate-monitoring-dataset-from-aeration-study-1.0.0\HRM_rawData\HRB"
hrb_data = load_hrb_data(subject=3, root_path=hrb_root)


def load_subject_info(info_path):
    """
    Load the subject-info.csv file.
    
    Parameters:
      info_path : Full file path to subject-info.csv.
      
    Returns:
      A DataFrame with the subject demographic and clinical data.
    """
    try:
        df = pd.read_csv(info_path)
        print(f"Loaded subject info with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading subject-info.csv: {e}")
        return None

# Example usage:
subject_info_path = "2_AI\respiratory-and-heart-rate-monitoring-dataset-from-aeration-study-1.0.0\subject-info.csv"
subject_info = load_subject_info(subject_info_path)


import os
import pandas as pd

def load_empatica_file(file_path):
    """
    Load an Empatica E4 data file.
    The first row contains the initial time (UTC) and the second row the sampling rate.
    The rest of the file contains the recorded data.
    """
    with open(file_path, 'r') as f:
        # Read meta information
        initial_time = f.readline().strip()
        sample_rate = f.readline().strip()
    # Now load the rest of the CSV data (assuming no header)
    df = pd.read_csv(file_path, skiprows=2, header=None)
    return initial_time, sample_rate, df

# Example usage:
participant_folder = "path/to/Wearable_Dataset/STRESS/S01"
acc_file = os.path.join(participant_folder, 'ACC.csv')
initial_time, sample_rate, acc_data = load_empatica_file(acc_file)
print("Session initial time (UTC):", initial_time)
print("Sample rate (Hz):", sample_rate)
print("ACC data shape:", acc_data.shape)

# Load stress levels – these files contain stage-by-stage self-reports.
stress_v1 = pd.read_csv("path/to/Wearable_Dataset/Stress_Level_v1.csv")
stress_v2 = pd.read_csv("path/to/Wearable_Dataset/Stress_Level_v2.csv")
print("Stress_Level_v1 shape:", stress_v1.shape)
print("Stress_Level_v2 shape:", stress_v2.shape)

# Load subject demographics.
subject_info = pd.read_csv("path/to/Wearable_Dataset/subject-info.csv")
print("Subject info shape:", subject_info.shape)






















def simulate_data(n_samples=200):
    """
    Simulate datasets for HRV, ECG, and behavioral metrics.
    
    HRV data includes features such as SDNN, RMSSD, and NN50.
    ECG data simulates features like amplitude and frequency parameters.
    Behavioral data includes simplified measures such as activity level and sleep quality.
    A binary stress label is derived from HRV values (e.g., low SDNN indicating higher stress).
    """
    np.random.seed(0)
    
    # Simulated HRV features
    hrv_data = pd.DataFrame({
        'SDNN': np.random.normal(50, 10, n_samples),
        'RMSSD': np.random.normal(40, 8, n_samples),
        'NN50': np.random.normal(5, 2, n_samples)
    })
    
    # Simulated ECG features
    ecg_data = pd.DataFrame({
        'ECG_Amplitude': np.random.normal(1.0, 0.2, n_samples),
        'ECG_Freq': np.random.normal(60, 5, n_samples)
    })
    
    # Simulated behavioral metrics
    behavior_data = pd.DataFrame({
        'Activity_Level': np.random.normal(3, 1, n_samples),
        'Sleep_Quality': np.random.normal(7, 1.5, n_samples)
    })
    
    # Derive stress label based on HRV metric (e.g., SDNN below threshold indicates stress)
    stress_label = (hrv_data['SDNN'] < 45).astype(int)
    
    return hrv_data, ecg_data, behavior_data, stress_label

def fuse_modalities(hrv_data, ecg_data, n_components=2):
    """
    Perform canonical correlation analysis (CCA) to fuse HRV and ECG data.
    
    CCA is used here as a proxy for more advanced probabilistic methods.
    The transformed features from both modalities are concatenated to form a fused feature space.
    """
    cca = CCA(n_components=n_components)
    hrv_transformed, ecg_transformed = cca.fit_transform(hrv_data, ecg_data)
    
    # Concatenating the transformed components into a single fused dataset
    fused_data = np.concatenate((hrv_transformed, ecg_transformed), axis=1)
    return fused_data, cca

def integrate_behavior(fused_data, behavior_data):
    """
    Integrate behavioral data with the fused HRV-ECG features.
    
    This simple concatenation reflects the idea of multimodal data fusion for holistic mental health monitoring.
    """
    # Concatenate along columns (axis=1)
    integrated_features = np.concatenate((fused_data, behavior_data.values), axis=1)
    return integrated_features

def predict_stress(features, stress_label):
    """
    Train a logistic regression classifier to predict stress from the integrated features.
    
    This predictive model represents a simple instantiation of the stress prediction paradigm.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(features, stress_label)
    predictions = model.predict(features)
    accuracy = accuracy_score(stress_label, predictions)
    
    return model, predictions, accuracy

if __name__ == "__main__":
    # Simulate data acquisition from multiple wearable modalities and behavioral inputs
    hrv_data, ecg_data, behavior_data, stress_label = simulate_data(n_samples=200)
    
    # Fuse HRV and ECG data using CCA (mimicking multimodal integration techniques)
    fused_data, cca_model = fuse_modalities(hrv_data, ecg_data, n_components=2)
    
    # Integrate fused data with additional behavioral metrics
    combined_features = integrate_behavior(fused_data, behavior_data)
    
    # Predict stress using the integrated feature set via logistic regression
    model, predictions, accuracy = predict_stress(combined_features, stress_label)
    
    # Output classification results
    print("Classification Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:\n", classification_report(stress_label, predictions))
    
    # Visualize the CCA-transformed feature space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(fused_data[:, 0], fused_data[:, 1], c=stress_label, cmap='viridis', label=stress_label)
    plt.title("CCA Fused Features (HRV and ECG)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label='Stress Label')
    plt.show()