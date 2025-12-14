import os
import tempfile
import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Paths
# ===============================
EMOTION_MODEL_PATH = "models/emotion_model.h5"
SPEAKER_MODEL_PATH = "models/speaker_identifier_model.h5"
SPEAKER_ENCODER_PATH = "models/speaker_identifier_encoder.pkl"

# ===============================
# Audio parameters (MATCH TRAINING)
# ===============================
SAMPLE_RATE = 22050

# Emotion
EM_N_MFCC = 40
EM_MAX_LEN = 174
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# Speaker
SP_DURATION = 3
SP_N_MFCC = 40
SP_MAX_LEN = 130

# ===============================
# Load models (cached)
# ===============================
@st.cache_resource
def load_models():
    emotion_model = load_model(EMOTION_MODEL_PATH)
    speaker_model = load_model(SPEAKER_MODEL_PATH)
    speaker_encoder = joblib.load(SPEAKER_ENCODER_PATH)
    return emotion_model, speaker_model, speaker_encoder

emotion_model, speaker_model, speaker_encoder = load_models()

# ===============================
# Emotion feature extraction
# ===============================
def extract_emotion_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=EM_N_MFCC).T

    if mfcc.shape[0] < EM_MAX_LEN:
        mfcc = np.pad(mfcc, ((0, EM_MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:EM_MAX_LEN]

    return mfcc

# ===============================
# Speaker feature extraction
# ===============================
def extract_speaker_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    target_len = SAMPLE_RATE * SP_DURATION
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = librosa.util.fix_length(y, size=target_len)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=SP_N_MFCC).T

    if mfcc.shape[0] < SP_MAX_LEN:
        mfcc = np.pad(mfcc, ((0, SP_MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:SP_MAX_LEN]

    return mfcc

# ===============================
# Predictions
# ===============================
def predict_emotion(file_path):
    mfcc = extract_emotion_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    probs = emotion_model.predict(mfcc, verbose=0)
    idx = np.argmax(probs)
    return EMOTION_LABELS[idx], float(probs[0][idx])

def predict_speaker(file_path):
    mfcc = extract_speaker_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)

    probs = speaker_model.predict(mfcc, verbose=0)
    idx = np.argmax(probs)
    raw_label = speaker_encoder.inverse_transform([idx])[0]

    speaker = f"Actor {int(raw_label)}"
    return speaker, float(probs[0][idx])

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
st.title("🎙 Speech Emotion & Speaker Recognition")

st.markdown(
    "Upload a speech audio file to **detect emotion** and **identify the speaker**."
)

uploaded = st.file_uploader("Upload WAV audio file", type=["wav"])

if uploaded:
    audio_path = save_uploaded_file(uploaded)
    st.audio(audio_path)

    if st.button("Analyze Audio"):
        with st.spinner("Processing audio..."):
            try:
                emotion, e_conf = predict_emotion(audio_path)
                speaker, s_conf = predict_speaker(audio_path)

                st.success("🎯 Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Emotion")
                    st.write(f"**Emotion:** {emotion}")
                    st.write(f"**Confidence:** {e_conf:.2f}")

                with col2:
                    st.subheader("Speaker")
                    st.write(f"**Speaker:** {speaker}")
                    st.write(f"**Confidence:** {s_conf:.2f}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Models trained on RAVDESS | CNN-LSTM for Emotion, CNN for Speaker Identification")

