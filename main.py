from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import os
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

# -----------------------------
# Loading Data
# -----------------------------
Strong_Flex_CF = pd.read_csv("data/Strong_Flex_CF.csv")
Strong_Flex_CF.head()

# -----------------------------
# Filtering Parameters
# -----------------------------
fs = 1000  # sampling rate in Hz
lowcut = 20
highcut = 450

# create bandpass filter
b, a = butter(N=4, Wn=[lowcut, highcut], btype="band", fs=fs)

# -----------------------------
# Functions
# -----------------------------
def clean_data(data):
    # center signal by removing mean
    mean_bicep = data.bicep.mean()
    normalised_bicep = np.array(data.bicep) - mean_bicep
    mean_tricep = data.tricep.mean()
    normalised_tricep = np.array(data.tricep) - mean_tricep

    # apply to signal
    filtered_bicep = filtfilt(b, a, normalised_bicep)
    filtered_tricep = filtfilt(b, a, normalised_tricep)

    # rectify signal
    rectified_bicep = np.abs(filtered_bicep)
    rectified_tricep = np.abs(filtered_tricep)

    return rectified_bicep, rectified_tricep


# -----------------------------
# Data Cleaning
# -----------------------------
cleaned_data_bicep, cleaned_data_tricep = clean_data(Strong_Flex_CF)

# -----------------------------
# Parameters
# -----------------------------
SAMPLING_RATE = 1000  # Hz
WINDOW_MS = 200  # window length in milliseconds
OVERLAP = 0.5  # 50% overlap

WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_MS / 1000)
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))

print("Window size (samples):", WINDOW_SIZE)
print("Stride:", STRIDE)

# Extract signals and labels
signals_bicep = cleaned_data_bicep
signals_tricep = cleaned_data_tricep
labels = Strong_Flex_CF.label


# -----------------------------
# Windowing function
# -----------------------------
def create_windows(signals, labels, window_size, stride):
    X = []
    y = []

    for start in range(0, len(signals) - window_size, stride):
        end = start + window_size

        window_signal = signals[start:end]
        window_labels = labels[start:end]

        # Only keep window if label is consistent
        if len(set(window_labels)) == 1:
            X.append(window_signal)
            y.append(
                window_labels.iloc[0]
            )  # Changed to .iloc[0] to get the first element by position

    return np.array(X), np.array(y)

# Plot cleaned signals

# Apply a moving average filter
window_size = 40  # Example window size, corresponds to 0.5 seconds at fs=200Hz

bicep_smoothed = pd.Series(signals_bicep).rolling(window=window_size).mean()

plt.figure(1)
plt.clf()
plt.plot(bicep_smoothed, label="Smoothed Bicep Signal")
plt.title('Rectified Bicep Signal')
plt.xlabel('x-axis (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show(block=False)

tricep_smoothed = pd.Series(signals_tricep).rolling(window=window_size).mean()

plt.figure(2)
plt.clf()
plt.plot(tricep_smoothed, label="Smoothed Tricep Signal")
plt.title('Rectified Tricep Signal')
plt.xlabel('x-axis (samples)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show(block=False)



# -----------------------------
# Create windows
# -----------------------------
X_bicep, y_bicep = create_windows(signals_bicep, labels, WINDOW_SIZE, STRIDE)
X_tricep, y_tricep = create_windows(signals_tricep, labels, WINDOW_SIZE, STRIDE)

# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

# Extracting features (RMS, MAV, WL, ZC, SSC)


def rms(signal):
    """Calculates the Root Mean Square of a signal."""
    signal = signal[~np.isnan(signal)]  # Remove NaN values
    if len(signal) == 0:
        return np.nan
    return np.sqrt(np.mean(signal**2))


def mav(signal):
    """Calculates the Mean Absolute Value of a signal."""
    signal = signal[~np.isnan(signal)]  # Remove NaN values
    if len(signal) == 0:
        return np.nan
    return np.mean(np.abs(signal))


def wl(signal):
    """Calculates the Waveform Length of a signal."""
    signal = signal[~np.isnan(signal)]  # Remove NaN values
    if len(signal) < 2:
        return np.nan
    return np.sum(np.abs(np.diff(signal)))


#def zc(signal, threshold=0):
    """Calculates the number of Zero Crossings in a signal."""
    signal = signal[~np.isnan(signal)]  # Remove NaN values
    if len(signal) < 2:
        return np.nan
    return np.sum(np.diff(np.sign(signal - threshold)) != 0)


def ssc(signal, threshold=0):
    """Calculates the number of Slope Sign Changes in a signal."""
    signal = signal[~np.isnan(signal)]  # Remove NaN values
    if len(signal) < 3:
        return np.nan
    return np.sum(
        (signal[1:-1] - signal[:-2]) * (signal[2:] - signal[1:-1]) < threshold
    )


print("Feature extraction functions defined.")

# Initialize a list to store features for each window
all_features = []

# Iterate through each window in X
for i in range(X_bicep.shape[0]):
    bicep_window_signal = X_bicep[i, :]
    tricep_window_signal = X_tricep[i, :]

    # Calculate features for the current window
    features_dict = {
        "RMS_1": rms(bicep_window_signal),
        "MAV_1": mav(bicep_window_signal),
        "WL_1": wl(bicep_window_signal),
        #"ZC_1": zc(bicep_window_signal),
        "SSC_1": ssc(bicep_window_signal),
        "RMS_2": rms(tricep_window_signal),
        "MAV_2": mav(tricep_window_signal),
        "WL_2": wl(tricep_window_signal),
        #"ZC_2": zc(tricep_window_signal),
        "SSC_2": ssc(tricep_window_signal),
    }
    all_features.append(features_dict)

# Create a DataFrame from the collected features
features_df = pd.DataFrame(all_features)

# Add the corresponding labels
features_df["Label"] = y_bicep
#print(y_bicep)
#print(y_tricep)

print(f"Generated features for {len(features_df)} windows.")
print("First 5 rows of the features DataFrame:")
print(features_df.head())
features_df.to_csv("data/features.csv", index=False)

# -----------------------------
# Normalization
# -----------------------------

for i in range(0, 8):  # Loop through feature columns (excluding the label)
    feature_name = features_df.columns[i]
    scaler = StandardScaler()
    features_df[feature_name] = scaler.fit_transform(
        features_df[[feature_name]]
    )  # Use double brackets to keep it as a DataFrame
    
features_df.to_csv("data/features_normalized.csv", index=False)

# -----------------------------
# Train-test split
# -----------------------------