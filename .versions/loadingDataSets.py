import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

#############################################
# 1. Empatica (Wearable_Dataset) Functions  #
#############################################
def load_empatica_file(file_path):
    """
    Load an Empatica E4 file.
    The first row contains the session start time (UTC) and the second row the sampling rate.
    The rest of the file contains the recorded data.
    """
    with open(file_path, 'r') as f:
        initial_time = f.readline().strip()        # initial time (UTC)
        sample_rate = f.readline().strip()           # sampling rate in Hz
    # Load remaining data; assume no header after skipping the first two rows
    df = pd.read_csv(file_path, skiprows=2, header=None)
    return initial_time, float(sample_rate), df

def load_wearable_stress_data(root_dir):
    """
    Load the Wearable_Dataset files.
    This function:
      - Loads self-reported stress levels (Stress_Level_v1.csv and Stress_Level_v2.csv)
      - Loads subject demographic info from subject-info.csv
      - Iterates through activity folders ('STRESS', 'AEROBIC', 'ANAEROBIC')
      - Loads each participant’s Empatica E4 files (ACC.csv, BVP.csv, EDA.csv, HR.csv, IBI.csv, tags.csv, TEMP.csv)
    """
    wearable_data = {}
    
    # Load stress level files
    try:
        stress_v1 = pd.read_csv(os.path.join(root_dir, "Stress_Level_v1.csv"))
        stress_v2 = pd.read_csv(os.path.join(root_dir, "Stress_Level_v2.csv"))
        wearable_data['stress_levels'] = {'v1': stress_v1, 'v2': stress_v2}
    except Exception as e:
        print("Error loading stress levels:", e)
    
    # Load subject demographics
    try:
        subj_info = pd.read_csv(os.path.join(root_dir, "subject-info.csv"))
        wearable_data['subject_info'] = subj_info
    except Exception as e:
        print("Error loading subject-info:", e)
    
    # Iterate through the three activity folders
    activities = ["STRESS", "AEROBIC", "ANAEROBIC"]
    wearable_data['activities'] = {}
    for activity in activities:
        activity_path = os.path.join(root_dir, activity)
        if os.path.exists(activity_path):
            part_list = [d for d in os.listdir(activity_path) if os.path.isdir(os.path.join(activity_path, d))]
            wearable_data['activities'][activity] = {}
            for participant in part_list:
                part_path = os.path.join(activity_path, participant)
                part_files = {}
                # Load each expected Empatica file
                for fname in ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv', 'TEMP.csv']:
                    fpath = os.path.join(part_path, fname)
                    if os.path.exists(fpath):
                        try:
                            # Use our empatica loader to get meta info and data
                            init_time, rate, df = load_empatica_file(fpath)
                            part_files[fname.split('.')[0]] = {'init_time': init_time, 'rate': rate, 'data': df}
                        except Exception as e:
                            print(f"Error loading {fname} for {participant}: {e}")
                wearable_data['activities'][activity][participant] = part_files
        else:
            print(f"Activity folder '{activity}' not found in {root_dir}.")
    return wearable_data

##################################################
# 2. BIDMC Dataset Functions (Respiratory Analysis) #
##################################################
def load_bidmc_csv(recording_dir, subject_num):
    """
    Load BIDMC CSV files for a given subject number.
    Expects files:
     - bidmc_##_Signals.csv
     - bidmc_##_Numerics.csv
     - bidmc_##_Breaths.csv
     - bidmc_##_Fix.txt (optional)
    """
    subject_str = f"{int(subject_num):02d}"
    signals_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Signals.csv")
    numerics_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Numerics.csv")
    breaths_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Breaths.csv")
    fix_file = os.path.join(recording_dir, f"bidmc_{subject_str}_Fix.txt")
    
    try:
        signals_df = pd.read_csv(signals_file)
    except Exception as e:
        print(f"Error loading {signals_file}: {e}")
        signals_df = None
    
    try:
        numerics_df = pd.read_csv(numerics_file)
    except Exception as e:
        print(f"Error loading {numerics_file}: {e}")
        numerics_df = None
        
    try:
        breaths_df = pd.read_csv(breaths_file)
    except Exception as e:
        print(f"Error loading {breaths_file}: {e}")
        breaths_df = None
    
    fix_params = None
    if os.path.exists(fix_file):
        with open(fix_file, 'r') as f:
            fix_params = f.read()
    
    return {'signals': signals_df, 'numerics': numerics_df, 'breaths': breaths_df, 'fixed': fix_params}

######################################################################
# 3. Dexcom G6 and Empatica E4 (Glycemic & Wearable) Data Functions    #
######################################################################
def load_dexcom_empatica(participant_dir):
    """
    Load data for a participant using Dexcom G6 and Empatica E4 wearable devices.
    Expected files: ACC.csv, BVP.csv, Dexcom.csv, EDA.csv, HR.csv, IBI.csv, TEMP.csv.
    """
    data = {}
    features = ['ACC.csv', 'BVP.csv', 'Dexcom.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']
    for f in features:
        fpath = os.path.join(participant_dir, f)
        try:
            df = pd.read_csv(fpath)
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            data[f.split('.')[0]] = df
        except Exception as e:
            print(f"Error loading {f} from {participant_dir}: {e}")
    # Load associated food log if available
    food_log_files = [f for f in os.listdir(participant_dir) if f.startswith("Food_Log")]
    if food_log_files:
        try:
            food_log_df = pd.read_csv(os.path.join(participant_dir, food_log_files[0]))
            data['Food_Log'] = food_log_df
        except Exception as e:
            print("Error loading Food Log:", e)
    return data

#########################################################
# 4. UT Dallas Non-EEG Data Functions (Stress & Relaxation) #
#########################################################
def load_non_eeg_utd_data(file_path):
    """
    Load non-EEG physiological data recorded at UT Dallas.
    This file is assumed to be continuous and includes a 'Timestamp' column.
    The data is segmented into experimental stages.
    """
    df = pd.read_csv(file_path)
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    experiment_start = df['Timestamp'].iloc[0]
    
    # Define the experimental stages and durations
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

    # Assign each timestamp a stage label
    def assign_stage(ts):
        for (start, end), (label, _) in zip(stage_boundaries, stages):
            if start <= ts < end:
                return label
        return "Unknown"
    
    df['Stage'] = df['Timestamp'].apply(assign_stage)
    return df

#############################################
# 5. Processed Pressure/EIT/ECG/PPG Data #
#############################################
def load_processed_dataset(subject, test_type, root_path):
    """
    Load a processed dataset CSV file for a given subject and test type.
    Expected filename format: ProcessedData_SubjectXX_testType.csv, where testType might be PEEP, PEEP_BH, or FEM.
    """
    subject_str = f"{int(subject):02d}"
    file_name = f"ProcessedData_Subject{subject_str}_{test_type}.csv"
    file_path = os.path.join(root_path, file_name)
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

#############################################
# 6. Subject Information Loader             #
#############################################
def load_subject_info(info_path):
    """
    Load subject-info.csv that contains demographic and clinical information.
    """
    try:
        df = pd.read_csv(info_path)
        return df
    except Exception as e:
        print(f"Error loading subject-info.csv: {e}")
        return None

#############################################
# 7. Integration: Combine All Datasets       #
#############################################
def integrate_data():
    """
    High-level function to load and combine data from each dataset.
    Modify the paths below as necessary.
    """
    # Define example paths (update these with your actual file locations)
    wearable_root = "wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.0"          
    bidmc_root = "bidmc-ppg-and-respiration-dataset-1.0.0"
    dexcom_empatica_root = "big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.2"          
    utd_non_eeg_file = "non-eeg-dataset-for-assessment-of-neurological-status-1.0.0"         
    processed_root = "path/to/ProcessedDataset"         
    subject_info_file = "path/to/subject-info.csv"      

    combined_data = {}
    
    # 1. Load Wearable Dataset (including stress level files and Empatica E4 data)
    print("Loading Wearable Dataset...")
    wearable = load_wearable_stress_data(wearable_root)
    combined_data["wearable"] = wearable
    
    # 2. Load BIDMC respiratory dataset (example for subject 1)
    print("Loading BIDMC dataset for subject 1...")
    bidmc = load_bidmc_csv(bidmc_root, subject_num=1)
    combined_data["bidmc"] = bidmc
    
    # 3. Load Dexcom G6 & Empatica E4 dataset (example for participant '001')
    print("Loading Dexcom/Empatica data for participant '001'...")
    dexcom_empatica = load_dexcom_empatica(os.path.join(dexcom_empatica_root, "001"))
    combined_data["dexcom_empatica"] = dexcom_empatica
    
    # 4. Load UT Dallas non-EEG data (with experimental stage labeling)
    print("Loading UT Dallas non-EEG data...")
    utd_data = load_non_eeg_utd_data(utd_non_eeg_file)
    combined_data["utd_non_eeg"] = utd_data
    
    # 5. Load Processed Dataset sample (subject 1, test type "PEEP")
    print("Loading Processed Dataset for subject 1 (PEEP test)...")
    processed = load_processed_dataset(subject=1, test_type="PEEP", root_path=processed_root)
    combined_data["processed"] = processed
    
    # 6. Load subject information (for processed data or general analysis)
    subj_info = load_subject_info(subject_info_file)
    combined_data["subject_info"] = subj_info
    
    return combined_data

#############################################
# 8. Main Execution                         #
#############################################
if __name__ == "__main__":
    # Build the integrated multimodal dataset:
    data_bundle = integrate_data()
    
    # Print a summary to verify successful integration:
    print("\n--- Integration Summary ---")
    if "wearable" in data_bundle and data_bundle["wearable"]:
        activities = data_bundle["wearable"].get("activities", {})
        for act, parts in activities.items():
            print(f"Activity '{act}': {list(parts.keys())} participants loaded")
    if data_bundle.get("bidmc") and data_bundle["bidmc"].get("signals") is not None:
        print("BIDMC signals shape:", data_bundle["bidmc"]["signals"].shape)
    if data_bundle.get("utd_non_eeg") is not None:
        print("UT Dallas non-EEG data samples with stage labeling:")
        print(data_bundle["utd_non_eeg"].head())
    if data_bundle.get("processed") is not None:
        print("Processed dataset sample shape:", data_bundle["processed"].shape if isinstance(data_bundle["processed"], pd.DataFrame) else "N/A")