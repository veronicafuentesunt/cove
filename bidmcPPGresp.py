import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert, detrend

def bandpass_filter(data, fs, lowcut, highcut, order=2):
    """
    Apply a Butterworth bandpass filter to the signal.

    Parameters:
      data: 1-D numpy array representing the signal.
      fs: Sampling frequency in Hz.
      lowcut: Lower cutoff frequency in Hz.
      highcut: Upper cutoff frequency in Hz.
      order: Filter order (default: 2).

    Returns:
      Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def estimate_respiratory_rate(ppg, fs, window_size_seconds=30, overlap_seconds=15,
                              resp_low=0.1, resp_high=0.5):
    """
    Estimate respiratory rate from a PPG signal.

    Parameters:
      ppg: 1-D numpy array of the PPG signal.
      fs: Sampling frequency in Hz.
      window_size_seconds: Length of the analysis window (seconds).
      overlap_seconds: Overlap between consecutive windows (seconds).
      resp_low: Lower bound of the respiratory frequency band (Hz).
      resp_high: Upper bound of the respiratory frequency band (Hz).

    Returns:
      times: List of time points (in seconds) for each estimation window.
      rates: List of estimated respiratory rates (in breaths per minute).
    """
    window_size = int(window_size_seconds * fs)
    overlap = int(overlap_seconds * fs)
    step = window_size - overlap
    rates = []
    times = []

    for start in range(0, len(ppg) - window_size, step):
        segment = ppg[start:start + window_size]
        # Remove baseline drift
        segment = detrend(segment)
        # Compute the analytic signal and obtain the amplitude envelope
        analytic_signal = hilbert(segment)
        envelope = np.abs(analytic_signal)
        # Filter the envelope to isolate the respiratory frequencies
        filtered_env = bandpass_filter(envelope, fs, resp_low, resp_high, order=2)

        # Compute FFT of the filtered envelope
        N = len(filtered_env)
        freqs = np.fft.rfftfreq(N, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(filtered_env))
        
        # Focus on frequencies within the respiratory band (0.1-0.5 Hz)
        resp_band = (freqs >= resp_low) & (freqs <= resp_high)
        if np.sum(resp_band) == 0:
            continue  # skip this window if no frequencies fall within range
        
        dominant_idx = np.argmax(fft_vals[resp_band])
        dominant_freq = freqs[resp_band][dominant_idx]
        rate_bpm = dominant_freq * 60  # convert Hz to breaths per minute
        
        rates.append(rate_bpm)
        times.append(start / fs)

    return times, rates

def process_subject(subject_id, data_dir, fs=125):
    """
    Process the Signals CSV file for a given subject.

    Parameters:
      subject_id: Subject number as an integer (e.g., 1 for bidmc_01_Signals.csv).
      data_dir: Directory containing the CSV files.
      fs: Sampling frequency in Hz.

    Returns:
      times: List of time points (in seconds) for each respiratory rate estimation.
      rates: List of estimated respiratory rates.
    """
    # Construct the file path to the subject's Signals CSV file, e.g., "bidmc_01_Signals.csv"
    signals_file = os.path.join(data_dir, f"bidmc_{subject_id:02d}_Signals.csv")

    try:
        df_signals = pd.read_csv(signals_file)
    except Exception as e:
        raise Exception(f"Error reading file {signals_file}: {e}")
    
    # Strip whitespace from the column headers so " PLETH" or "PLETH " become "PLETH"
    df_signals.columns = df_signals.columns.str.strip()
    
    # Check for the existence of the "PLETH" column
    if "PLETH" not in df_signals.columns:
        # Print available columns for debugging
        print(f"Available columns in {signals_file}: {df_signals.columns.tolist()}")
        raise ValueError(f"'PLETH' column not found in {signals_file}. Check CSV headers.")

    # Extract the PPG signal from the "PLETH" column
    ppg = df_signals["PLETH"].values

    return estimate_respiratory_rate(ppg, fs)

# --- Main script for processing all subjects ---
if __name__ == '__main__':
    fs = 125  # Sampling frequency in Hz (as in the BIDMC dataset)
    data_dir = r"bidmc-ppg-and-resp\bidmc_csv"  # Replace with the folder path where your CSV files are stored

    all_subjects_rates = {}

    # Loop through subjects 1 to 53
    for subject in range(1, 54):
        try:
            times, rates = process_subject(subject, data_dir, fs)
            all_subjects_rates[subject] = (times, rates)
            print(f"Processed subject {subject:02d}: {len(rates)} respiratory rate estimates.")
        except Exception as e:
            print(f"Error processing subject {subject:02d}: {e}")

    # Plot the estimated respiratory rates for each subject using subplots
    num_subjects = len(all_subjects_rates)
    cols = 5
    rows = (num_subjects // cols) + (1 if num_subjects % cols else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), squeeze=False)
    fig.suptitle("Estimated Respiratory Rate Over Time for Each Subject", fontsize=16)

    subject_idx = 0
    for subject, (times, rates) in all_subjects_rates.items():
        row = subject_idx // cols
        col = subject_idx % cols
        ax = axes[row, col]
        ax.plot(times, rates, marker='o', linestyle='-', color='blue')
        ax.set_title(f"Subject {subject:02d}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Respiratory Rate (BPM)")
        ax.grid(True)
        subject_idx += 1

    # Remove any unused subplots
    for i in range(subject_idx, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()