
import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer


# =========================================================
# CUSTOM ATTENTION LAYER
# =========================================================

class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):

        score = tf.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = tf.reduce_sum(
            inputs * attention_weights,
            axis=1
        )

        return context_vector


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🎙️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎙️ EmotionSense AI")

st.subheader(
    "Deep Speech Emotion Recognition & Explainable Audio Intelligence System"
)

st.write(
    "Upload a speech audio file and the AI model will predict "
    "the speaker's emotional state."
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "models/emotionsense_best.keras"
MEAN_PATH = "models/train_mean.npy"
STD_PATH = "models/train_std.npy"
LABEL_PATH = "models/label_classes.npy"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_emotion_model():

    model = load_model(
        MODEL_PATH,
        custom_objects={
            "AttentionLayer": AttentionLayer
        }
    )

    return model


@st.cache_resource
def load_normalization():

    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    labels = np.load(
        LABEL_PATH,
        allow_pickle=True
    )

    return mean, std, labels


# =========================================================
# AUDIO PREPROCESSING
# =========================================================

TARGET_SR = 16000
TARGET_DURATION = 3.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)


def preprocess_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=30
    )

    max_amplitude = np.max(
        np.abs(audio_trimmed)
    )

    if max_amplitude > 0:

        audio_trimmed = (
            audio_trimmed / max_amplitude
        )

    if len(audio_trimmed) < TARGET_LENGTH:

        audio_processed = np.pad(
            audio_trimmed,
            (
                0,
                TARGET_LENGTH - len(audio_trimmed)
            ),
            mode="constant"
        )

    else:

        start = (
            len(audio_trimmed) - TARGET_LENGTH
        ) // 2

        audio_processed = audio_trimmed[
            start:start + TARGET_LENGTH
        ]

    return audio_processed.astype(np.float32), sr


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(audio, sr=16000):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=64
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_chroma=12
    )

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_bands=6
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    rms = librosa.feature.rms(
        y=audio
    )

    return {
        "mfcc": mfcc,
        "mel": mel_db,
        "chroma": chroma,
        "contrast": contrast,
        "zcr": zcr,
        "rms": rms
    }


# =========================================================
# FEATURE FUSION
# =========================================================

def fuse_features(features):

    mfcc = features["mfcc"]
    mel = features["mel"]
    chroma = features["chroma"]
    contrast = features["contrast"]
    zcr = features["zcr"]
    rms = features["rms"]

    min_frames = min(
        mfcc.shape[1],
        mel.shape[1],
        chroma.shape[1],
        contrast.shape[1],
        zcr.shape[1],
        rms.shape[1]
    )

    mfcc = mfcc[:, :min_frames]
    mel = mel[:, :min_frames]
    chroma = chroma[:, :min_frames]
    contrast = contrast[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]

    fused = np.concatenate(
        [
            mfcc,
            mel,
            chroma,
            contrast,
            zcr,
            rms
        ],
        axis=0
    )

    return fused.astype(np.float32)


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    temp_path = "/tmp/input.wav"

    with open(temp_path, "wb") as f:

        f.write(
            uploaded_file.getbuffer()
        )

    if st.button("🔍 Analyze Emotion"):

        with st.spinner(
            "EmotionSense AI is analyzing the speech..."
        ):

            model = load_emotion_model()

            saved_mean, saved_std, class_names = (
                load_normalization()
            )

            audio, sr = preprocess_audio(
                temp_path
            )

            features = extract_features(
                audio,
                sr
            )

            fused = fuse_features(
                features
            )

            fused = fused.T

            fused = np.expand_dims(
                fused,
                axis=0
            )

            fused_normalized = (
                fused - saved_mean
            ) / saved_std

            probabilities = model.predict(
                fused_normalized,
                verbose=0
            )[0]

            predicted_index = np.argmax(
                probabilities
            )

            predicted_emotion = (
                class_names[predicted_index]
            )

            confidence = (
                probabilities[predicted_index]
            )

            top_indices = np.argsort(
                probabilities
            )[-3:][::-1]


        # =================================================
        # RESULT
        # =================================================

        st.success(
            f"Predicted Emotion: {predicted_emotion.upper()}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Emotion",
                predicted_emotion.upper()
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # =================================================
        # TOP 3 PREDICTIONS
        # =================================================

        st.subheader(
            "📊 Top 3 Emotion Predictions"
        )

        for rank, idx in enumerate(
            top_indices,
            start=1
        ):

            emotion = class_names[idx]

            probability = (
                probabilities[idx] * 100
            )

            st.write(
                f"**{rank}. {emotion.upper()}**"
            )

            st.progress(
                float(probabilities[idx])
            )

            st.write(
                f"{probability:.2f}%"
            )


        # =================================================
        # TECHNICAL INFORMATION
        # =================================================

        with st.expander(
            "🔬 Technical Model Information"
        ):

            st.write(
                "Architecture: CNN + BiLSTM + Temporal Attention"
            )

            st.write(
                "Sampling Rate: 16 kHz"
            )

            st.write(
                "Input Duration: 3 seconds"
            )

            st.write(
                "Feature Dimensions: 125"
            )

            st.write(
                "Temporal Steps: 94"
            )

            st.write(
                "Output Classes: 8"
            )

            st.write(
                "Feature Fusion: MFCC + Mel Spectrogram + "
                "Chroma + Spectral Contrast + ZCR + RMS"
            )
