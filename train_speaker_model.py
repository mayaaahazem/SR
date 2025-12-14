import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# ===============================
# Paths
# ===============================
DATA_PATH = "data/cleaned_data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "speaker_identifier_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "speaker_identifier_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Parameters
# ===============================
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 174   # ~4 seconds

# ===============================
# Feature Extraction
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

# ===============================
# Load Data
# ===============================
X, y = [], []

for file in os.listdir(DATA_PATH):
    if file.endswith(".wav"):
        path = os.path.join(DATA_PATH, file)

        # Speaker ID from filename
        speaker_id = file.split("-")[6].replace(".wav", "")
        features = extract_mfcc(path)

        X.append(features)
        y.append(speaker_id)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("Loaded data")
print("X shape:", X.shape)
print("y samples:", len(y))

# ===============================
# Encode Speaker Labels (24 speakers)
# ===============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, ENCODER_PATH)
print(f" Speaker encoder saved to {ENCODER_PATH}")
print("Speakers:", encoder.classes_)

# ===============================
# Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ===============================
# Build CNN + LSTM Model
# ===============================
model = Sequential([
    Conv1D(64, 5, activation="relu", input_shape=X_train.shape[1:]),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    LSTM(128),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# Train
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    verbose=1
)

# ===============================
# Save Model
# ===============================
model.save(MODEL_PATH)
print(f"Speaker identification model saved to {MODEL_PATH}")
