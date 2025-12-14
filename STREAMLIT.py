import os
import tempfile
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Paths (same as CLI)
# ===============================
MODEL_PATH = "models/speaker_identifier_model.h5"
ENCODER_PATH = "models/speaker_identifier_encoder.pkl"

# ===============================
# Audio parameters (MUST match training)
# ===============================
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130

# ===============================
# Load model and encoder (once)
# ===============================
@st.cache_resource
def load_speaker_model():
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

model, encoder = load_speaker_model()

# ===============================
# Feature extraction (MATCHES CLI)
# ===============================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    target_len = SAMPLE_RATE * DURATION
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = librosa.util.fix_length(y, size=target_len)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc

# ===============================
# Predict speaker
# ===============================
def predict_speaker(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    probs = model.predict(mfcc, verbose=0)
    idx = np.argmax(probs)
    confidence = float(probs[0][idx])

    raw_label = encoder.inverse_transform([idx])[0]
    speaker = f"Actor {int(raw_label)}"   # ✔ Actor 8 not 08

    return speaker, confidence

# ===============================
# Save uploaded file
# ===============================
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

# ===============================
# Streamlit UI
# ===============================
st.title("🎙 Speaker Identification System")
st.markdown("Upload a speech audio file to identify the speaker.")

uploaded = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded is not None:
    temp_audio = save_uploaded_file(uploaded)

    st.audio(temp_audio)

    if st.button("Identify Speaker"):
        with st.spinner("Analyzing speaker..."):
            try:
                speaker, confidence = predict_speaker(temp_audio)
                st.success("🎯 Prediction Result")
                st.markdown(f"**Speaker:** {speaker}")
                st.markdown(f"**Confidence:** {confidence:.2f}")
            except Exception as e:
                st.error(f"Speaker prediction failed: {e}")

st.markdown("---")
st.caption("Model: CNN-based Speaker Identifier | Dataset: RAVDESS")

