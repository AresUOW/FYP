from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

# -----------------------------
# Loading Data
# -----------------------------
Strong_Flex_CF = pd.read_csv('data/Strong_Flex_CF.csv')

# -----------------------------
# Functions
# -----------------------------

def clean_data(data):
    mean = data.mean()
    normalised = np.array(data.Bicep) - mean
    
    fs = 1000  # sampling rate in Hz
    lowcut = 20
    highcut = 450

    # create bandpass filter
    b, a = butter(N=4, Wn=[lowcut, highcut], btype='band', fs=fs)
    
    # apply to signal
    filtered = filtfilt(b, a, normalised)
    
# -----------------------------
# Data Cleaning
# -----------------------------


# -----------------------------
# Parameters
# -----------------------------
SAMPLING_RATE = 1000        # Hz
WINDOW_MS = 200             # window length in milliseconds
OVERLAP = 0.5               # 50% overlap

WINDOW_SIZE = int(SAMPLING_RATE * WINDOW_MS / 1000)
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))

print("Window size (samples):", WINDOW_SIZE)
print("Stride:", STRIDE)

# Extract signals and labels
signals = bicep_smoothed
labels = flexed_data.Label

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
            y.append(window_labels.iloc[0]) # Changed to .iloc[0] to get the first element by position

    return np.array(X), np.array(y)

# -----------------------------
# Create windows
# -----------------------------
X, y = create_windows(signals, labels, WINDOW_SIZE, STRIDE)

#print("Shape of X:", X.shape)
#print("Shape of y:", y.shape)

#Extracting features (RMS, MAV, WL, ZC, SSC)

features = ['RMS', 'MAV', 'WL', 'ZC', 'SSC']

for feature in features:

    def rms(signal):
        """Calculates the Root Mean Square of a signal."""
        signal = signal[~np.isnan(signal)] # Remove NaN values
        if len(signal) == 0: return np.nan
        return np.sqrt(np.mean(signal**2))

    def mav(signal):
        """Calculates the Mean Absolute Value of a signal."""
        signal = signal[~np.isnan(signal)] # Remove NaN values
        if len(signal) == 0: return np.nan
        return np.mean(np.abs(signal))

    def wl(signal):
        """Calculates the Waveform Length of a signal."""
        signal = signal[~np.isnan(signal)] # Remove NaN values
        if len(signal) < 2: return np.nan
        return np.sum(np.abs(np.diff(signal)))

    def zc(signal, threshold=0):
        """Calculates the number of Zero Crossings in a signal."""
        signal = signal[~np.isnan(signal)] # Remove NaN values
        if len(signal) < 2: return np.nan
        return np.sum(np.diff(np.sign(signal - threshold)) != 0)

    def ssc(signal, threshold=0):
        """Calculates the number of Slope Sign Changes in a signal."""
        signal = signal[~np.isnan(signal)] # Remove NaN values
        if len(signal) < 3: return np.nan
        return np.sum((signal[1:-1] - signal[:-2]) * (signal[2:] - signal[1:-1]) < threshold)

    print("Feature extraction functions defined.")

# Initialize a list to store features for each window
all_features = []

# Iterate through each window in X
for i in range(X.shape[0]):
    window_signal = X[i, :]

    # Calculate features for the current window
    features_dict = {
        'RMS': rms(window_signal),
        'MAV': mav(window_signal),
        'WL': wl(window_signal),
        'ZC': zc(window_signal),
        'SSC': ssc(window_signal)
    }
    all_features.append(features_dict)

# Create a DataFrame from the collected features
features_df = pd.DataFrame(all_features)

# Add the corresponding labels
features_df['Label'] = y

print(f"Generated features for {len(features_df)} windows.")
print("First 5 rows of the features DataFrame:")
display(features_df.head())

