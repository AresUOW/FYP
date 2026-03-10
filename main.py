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
Strong_Straight_CF = pd.read_csv("data/Strong_Straight_CF.csv")
dataset = pd.concat([Strong_Flex_CF, Strong_Straight_CF], ignore_index=True)
#Strong_Flex_CF.head()

# Create label mapping function
def map_labels_to_integers(labels):
    """
    Maps string labels to integers.
    Returns the mapped labels and the label dictionary for reference.
    """
    unique_labels = labels.unique()
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = labels.map(label_to_int)
    
    print("Label Mapping:")
    for label, idx in label_to_int.items():
        print(f"  {label} -> {idx}")
    
    return int_labels, label_to_int

# Apply label mapping
dataset['label'], label_mapping = map_labels_to_integers(dataset['label'])

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
cleaned_data_bicep, cleaned_data_tricep = clean_data(dataset)

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
labels = dataset.label


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
plt.title("Rectified Bicep Signal")
plt.xlabel("x-axis (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show(block=False)

tricep_smoothed = pd.Series(signals_tricep).rolling(window=window_size).mean()

plt.figure(2)
plt.clf()
plt.plot(tricep_smoothed, label="Smoothed Tricep Signal")
plt.title("Rectified Tricep Signal")
plt.xlabel("x-axis (samples)")
plt.ylabel("Amplitude")
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

    # def zc(signal, threshold=0):
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
        # "ZC_1": zc(bicep_window_signal),
        "SSC_1": ssc(bicep_window_signal),
        "RMS_2": rms(tricep_window_signal),
        "MAV_2": mav(tricep_window_signal),
        "WL_2": wl(tricep_window_signal),
        # "ZC_2": zc(tricep_window_signal),
        "SSC_2": ssc(tricep_window_signal),
    }
    all_features.append(features_dict)

# Create a DataFrame from the collected features
features_df = pd.DataFrame(all_features)

# Add the corresponding labels
features_df["Label"] = y_bicep
# print(y_bicep)
# print(y_tricep)

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

# Prepare for the LSTM model
training_data_len = int(len(features_df) * 0.8)

# Preprocessing Stages
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df.drop("Label", axis=1))

training_data = scaled_features[
    :training_data_len
]  # Use the first 80% of the data for training
test_data = scaled_features[training_data_len:]

#print("Training data:", training_data)

# Create a sliding window of 60 time steps for the LSTM input
timesteps = 5
X_train, y_train = [], []

# Use y_bicep[:training_data_len] to get labels aligned with training_data
training_labels = y_bicep[:training_data_len]
#print("Training labels:", training_labels)
print("Length of training data:", len(training_data))

for i in range(len(training_data) - timesteps):
    X_train.append(training_data[i:i + timesteps])
    y_train.append(training_labels[i + timesteps])

X_train = np.array(X_train)
y_train = np.array(y_train)

# One-hot encode labels for multi-class classification
num_classes = len(label_mapping)
y_train_encoded = keras.utils.to_categorical(y_train, num_classes=num_classes)

print(f"Number of classes: {num_classes}")
print(f"Shape of y_train_encoded: {y_train_encoded.shape}")

# Building the Model
model = keras.models.Sequential(
    [
        keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[keras.metrics.Accuracy()])

training = model.fit(X_train, y_train_encoded, batch_size=64, epochs=30, validation_split=0.2)

# Prep the test data
test_data = scaled_features[training_data_len:]
test_labels = y_bicep[training_data_len:]

X_test, y_test = [], []

for i in range(len(test_data) - timesteps):
    X_test.append(test_data[i:i + timesteps])
    y_test.append(test_labels[i + timesteps])

X_test = np.array(X_test)
y_test = np.array(y_test)
print("First 5 samples of X_test:")
print(X_test[:5])
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print("Test labels:", y_test[:5])

# Make a Prediction
predictions = model.predict(X_test)
print("Shape of predictions:", predictions.shape)
print("First 5 raw predictions:\n", predictions[:5])

# Convert probabilities to class labels using argmax
predicted_labels = np.argmax(predictions, axis=1)
print("\nFirst 5 predicted labels:", predicted_labels[:5])
print("First 5 actual labels:", y_test[:5])

# Calculate accuracy and other metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, predicted_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_labels))
print("\nClassification Report:")
# Create reverse mapping for labels and get all unique classes
int_to_label = {v: k for k, v in label_mapping.items()}
unique_classes = sorted(np.unique(np.concatenate([y_test, predicted_labels])))
label_names = [int_to_label.get(i, str(i)) for i in unique_classes]
print(classification_report(y_test, predicted_labels, labels=unique_classes, target_names=label_names))

# Plot Data
train = all_features[:training_data_len]
test = all_features[training_data_len:]

test = test.copy()
# test["Predictions"] = predictions

# plt.figure(figsize=(12,8))
# plt.plot(train['date'], train['label'], label="Train (Actual)", color='blue')
# plt.plot(test['date'], test['label'], label="Test (Actual)", color='orange')
# plt.plot(test['date'], test['Predictions'], label="Predictions", color='red')
# plt.title("Our Stock Predictions")
# plt.xlabel("Date")
# plt.ylabel("Close Price")
# plt.legend()
# plt.show()