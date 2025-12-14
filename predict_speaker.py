import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Paths
# ===============================
MODEL_PATH = "models/speaker_identifier_model.h5"
ENCODER_PATH = "models/speaker_identifier_encoder.pkl"

# ===============================
# Audio parameters (MUST match training)
# ===============================
SAMPLE_RATE = 22050
DURATION = 3          # seconds
N_MFCC = 40
MAX_LEN = 130         # adjust if different in training

# ===============================
# Load model and encoder
# ===============================
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

# ===============================
# Feature extraction
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    target_len = SAMPLE_RATE * DURATION
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = librosa.util.fix_length(y, size=target_len)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

# ===============================
# Record from microphone
# ===============================
def record_audio(output_file):
    print(f"\n🎙 Recording for {DURATION} seconds...")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )
    sd.wait()
    sf.write(output_file, audio, SAMPLE_RATE)
    print(f"✅ Saved recording to {output_file}")

# ===============================
# Predict speaker
# ===============================
def predict_speaker(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    probs = model.predict(mfcc)
    idx = np.argmax(probs)
    confidence = probs[0][idx]

    raw_label = encoder.inverse_transform([idx])[0]
    speaker = f"Actor {int(raw_label)}"   # 🔥 FIX HERE

    return speaker, confidence

# ===============================
# Main
# ===============================
if __name__ == "__main__":

    choice = input("\nUse Microphone or File? (m/f): ").strip().lower()

    if choice == "m":
        test_file = "temp_mic.wav"
        record_audio(test_file)

    elif choice == "f":
        test_file = input("Enter path to .wav file: ").strip()
        if not os.path.exists(test_file):
            print("❌ File not found")
            exit()

    else:
        print("❌ Invalid choice")
        exit()

    speaker, confidence = predict_speaker(test_file)

    print("\n🎯 Speaker Identification Result")
    print(f"Speaker    : {speaker}")
    print(f"Confidence : {confidence:.2f}")
