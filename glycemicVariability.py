import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Data Loading Functions
# --------------------------

def load_csv_with_timestamp(file_path, time_col, delimiter=',', dtype=None, chunksize=10**5, rename_map=None):
    """
    Loads a CSV file in chunks, explicitly parsing the given time column, and optionally renames columns.
    
    Parameters:
      file_path (str): Path to the CSV file.
      time_col (str): Name of the datetime column to parse.
      delimiter (str): CSV delimiter.
      dtype (dict): Dictionary specifying data types for columns to reduce memory usage.
      chunksize (int): Number of rows per chunk.
      rename_map (dict): Optional dictionary for renaming columns after loading.
      
    Returns:
      A pandas DataFrame with the time column parsed (and columns renamed), or None if there's an error.
    """
    try:
        chunks = []
        for chunk in pd.read_csv(file_path, delimiter=delimiter, low_memory=True,
                                 dtype=dtype, chunksize=chunksize, parse_dates=[time_col]):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        if rename_map is not None:
            df = df.rename(columns=rename_map)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_subject_data(subject_folder, subject_id):
    """
    Loads all expected feature CSV files for one subject.
    
    Expected files for subject e.g., "001":
      ACC_001.csv, BVP_001.csv, Dexcom_001.csv, EDA_001.csv, HR_001.csv, IBI_001.csv,
      TEMP_001.csv, and Food_Log_001.csv.
      
    For each type:
      - ACC: Expects columns "datetime", "acc_x", "acc_y", "acc_z".
      - BVP: Expects columns "datetime", "bvp".
      - Dexcom: Expects several columns; here, we parse "Timestamp (YYYY-MM-DDThh:mm:ss)",
                then rename it to "datetime" and "Glucose Value (mg/dL)" to "glucose".
      - EDA: Expects columns "datetime", "eda".
      - HR: Expects columns "datetime", "hr".
      - IBI: Expects columns "datetime", "ibi".
      - TEMP: Expects columns "datetime", "temp".
      - Food_Log: Is read without a datetime conversion.
      
    Returns:
      A dictionary mapping each feature name to its corresponding DataFrame.
    """
    data = {}
    features = ["ACC", "BVP", "Dexcom", "EDA", "HR", "IBI", "TEMP"]
    
    # Define dtype specifications to reduce memory usage.
    dtype_specs = {
        "ACC": {"acc_x": np.float32, "acc_y": np.float32, "acc_z": np.float32},
        "BVP": {"bvp": np.float32},
        "Dexcom": {"Glucose Value (mg/dL)": np.float32},  # Only converting the glucose value
        "EDA": {"eda": np.float32},
        "HR": {"hr": np.float32},
        "IBI": {"ibi": np.float32},
        "TEMP": {"temp": np.float32}
    }
    
    for feature in features:
        file_name = f"{feature}_{subject_id}.csv"
        file_path = os.path.join(subject_folder, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            data[feature] = None
        else:
            # Set parameters for each file type.
            if feature == "ACC":
                time_column = "datetime"
                rename_map = None
            elif feature == "BVP":
                time_column = "datetime"
                rename_map = None
            elif feature == "Dexcom":
                time_column = "Timestamp (YYYY-MM-DDThh:mm:ss)"
                rename_map = {"Timestamp (YYYY-MM-DDThh:mm:ss)": "datetime",
                              "Glucose Value (mg/dL)": "glucose"}
            elif feature == "EDA":
                time_column = "datetime"
                rename_map = None
            elif feature == "HR":
                time_column = "datetime"
                rename_map = None
            elif feature == "IBI":
                time_column = "datetime"
                rename_map = None
            elif feature == "TEMP":
                time_column = "datetime"
                rename_map = None
            else:
                time_column = "Timestamp"
                rename_map = None

            file_dtype = dtype_specs.get(feature, None)
            df = load_csv_with_timestamp(
                file_path,
                time_col=time_column,
                dtype=file_dtype,
                chunksize=10**5,
                rename_map=rename_map
            )
            data[feature] = df

    # Load the Food_Log file without datetime parsing.
    food_log_file = os.path.join(subject_folder, f"Food_Log_{subject_id}.csv")
    if not os.path.exists(food_log_file):
        print(f"Warning: {food_log_file} does not exist.")
        data["Food_Log"] = None
    else:
        try:
            food_df = pd.read_csv(food_log_file)
        except Exception as e:
            print(f"Error reading {food_log_file}: {e}")
            food_df = None
        data["Food_Log"] = food_df

    return data

def load_all_subjects(root_data_dir):
    """
    Loads data for all subjects from the root directory.
    
    The root directory should contain subfolders labeled "001" to "016".
    
    Returns:
      A dictionary mapping subject IDs (as zero-padded strings) to their data dictionaries.
    """
    subjects_data = {}
    for sub_id in range(1, 17):
        subject_id = f"{sub_id:03d}"
        subject_folder = os.path.join(root_data_dir, subject_id)
        if os.path.isdir(subject_folder):
            subjects_data[subject_id] = load_subject_data(subject_folder, subject_id)
        else:
            print(f"Warning: Subject folder {subject_folder} not found.")
    return subjects_data

def load_demographics(root_data_dir):
    """
    Loads the Demographics.csv file from the root directory.
    
    Returns:
      A DataFrame containing demographic information.
    """
    demo_file = os.path.join(root_data_dir, "Demographics.csv")
    try:
        demo_df = pd.read_csv(demo_file)
    except Exception as e:
        print(f"Error reading {demo_file}: {e}")
        demo_df = None
    return demo_df

# --------------------------
# Analysis Functions
# --------------------------

def analyze_correlations(subject_data, resample_interval="5T"):
    """
    Resamples and merges the Dexcom (glucose), HR, BVP, and TEMP data
    to the given time interval and computes the correlations between:
      - Glucose vs. HR and BVP.
      - TEMP vs. HR and BVP.
      
    Parameters:
      subject_data (dict): The dictionary of DataFrames for a single subject.
      resample_interval (str): The resampling frequency (e.g., "5T" for 5 minutes).
      
    Returns:
      A merged DataFrame and prints scatter plots and correlation values.
    """
    # Extract the required DataFrames:
    glucose_df = subject_data["Dexcom"]
    hr_df = subject_data["HR"]
    bvp_df = subject_data["BVP"]
    temp_df = subject_data["TEMP"]
    
    # Set datetime as index if not already (assuming data was parsed properly during loading)
    glucose_df = glucose_df.set_index("datetime")
    hr_df = hr_df.set_index("datetime")
    bvp_df = bvp_df.set_index("datetime")
    temp_df = temp_df.set_index("datetime")
    
    # Resample each signal to a common time interval.
    # Dexcom data is likely recorded every 5 minutes, so we use a 5-minute resampling.
    glucose_resampled = glucose_df.resample(resample_interval).mean()
    hr_resampled = hr_df.resample(resample_interval).mean()
    bvp_resampled = bvp_df.resample(resample_interval).mean()
    temp_resampled = temp_df.resample(resample_interval).mean()
    
    # Merge the resampled data on the datetime index.
    merged_df = pd.concat([
        glucose_resampled["glucose"],
        hr_resampled["hr"],
        bvp_resampled["bvp"],
        temp_resampled["temp"]
    ], axis=1)
    
    # Drop rows where glucose data is missing (or you could drop any row with missing values).
    merged_df = merged_df.dropna(subset=["glucose"])
    
    print("Merged Data Sample:")
    print(merged_df.head())
    
    # Plot relationships:
    # Glucose vs. HR and BVP
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    plt.scatter(merged_df["glucose"], merged_df["hr"], alpha=0.6)
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Glucose vs. Heart Rate")
    
    plt.subplot(1,2,2)
    plt.scatter(merged_df["glucose"], merged_df["bvp"], alpha=0.6, color='green')
    plt.xlabel("Glucose (mg/dL)")
    plt.ylabel("Blood Volume Pulse")
    plt.title("Glucose vs. BVP")
    plt.tight_layout()
    plt.show()
    
    # TEMP vs. HR and BVP
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    plt.scatter(merged_df["temp"], merged_df["hr"], alpha=0.6, color='orange')
    plt.xlabel("Skin Temperature")
    plt.ylabel("Heart Rate (bpm)")
    plt.title("Temp vs. Heart Rate")
    
    plt.subplot(1,2,2)
    plt.scatter(merged_df["temp"], merged_df["bvp"], alpha=0.6, color='purple')
    plt.xlabel("Skin Temperature")
    plt.ylabel("Blood Volume Pulse")
    plt.title("Temp vs. BVP")
    plt.tight_layout()
    plt.show()
    
    # Compute correlation coefficients:
    corr_glucose_hr = merged_df["glucose"].corr(merged_df["hr"])
    corr_glucose_bvp = merged_df["glucose"].corr(merged_df["bvp"])
    corr_temp_hr = merged_df["temp"].corr(merged_df["hr"])
    corr_temp_bvp = merged_df["temp"].corr(merged_df["bvp"])
    
    print("Correlation between Glucose and HR:", corr_glucose_hr)
    print("Correlation between Glucose and BVP:", corr_glucose_bvp)
    print("Correlation between TEMP and HR:", corr_temp_hr)
    print("Correlation between TEMP and BVP:", corr_temp_bvp)
    
    print("Full Correlation Matrix:")
    print(merged_df.corr())
    
    return merged_df

# --------------------------
# Main Script
# --------------------------

if __name__ == '__main__':
    # Update this path to your dataset's root directory.
    root_data_dir = r"glycemic-variability"  # REPLACE with your actual path
    
    # Load data for all subjects.
    all_subjects_data = load_all_subjects(root_data_dir)
    
    # Load demographics data from the root folder.
    demographics = load_demographics(root_data_dir)
    
    # For example, focus our analysis on subject "008"
    subject_id = "008"
    if subject_id in all_subjects_data:
        print(f"Data summary for subject {subject_id}:")
        subject_data = all_subjects_data[subject_id]
        for feature, df in subject_data.items():
            if df is not None:
                print(f"  {feature} data shape: {df.shape}")
            else:
                print(f"  {feature} data: Not available")
        
        # Perform correlation analysis to see how glucose affects HR and BVP,
        # and the relationship between TEMP, HR, and BVP.
        merged_data = analyze_correlations(subject_data, resample_interval="5T")
    else:
        print(f"Subject {subject_id} data not found.")