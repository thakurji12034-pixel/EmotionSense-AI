
# EmotionSense AI

### Deep Speech Emotion Recognition & Explainable Audio Intelligence System

EmotionSense AI is a deep learning based Speech Emotion Recognition (SER) system that analyzes human speech and predicts the emotional state from an uploaded WAV audio file.

The system combines advanced audio feature engineering with a hybrid deep learning architecture consisting of:

- CNN for local acoustic pattern extraction
- Bidirectional LSTM for temporal sequence learning
- Temporal Attention for important time-region weighting
- Explainable AI through attention analysis
- Speaker-independent evaluation
- Streamlit web interface for real-time prediction

---

## Features

- 🎙️ Speech emotion recognition from WAV files
- 🧠 CNN + BiLSTM + Attention architecture
- 🔊 16 kHz mono audio preprocessing
- 🎵 MFCC, Mel Spectrogram, Chroma and Spectral Contrast features
- 📊 Confidence scores and Top-3 predictions
- 🔍 Attention-based explainability
- 👤 Speaker-independent train/validation/test split
- 🌐 Streamlit-based interactive application

---

## Emotions

The model recognizes 8 emotions:

1. Angry
2. Calm
3. Disgust
4. Fearful
5. Happy
6. Neutral
7. Sad
8. Surprised

---

## Model Architecture

```text
Input Speech
     ↓
Audio Preprocessing
     ↓
Feature Extraction
     ├── MFCC
     ├── Mel Spectrogram
     ├── Chroma
     ├── Spectral Contrast
     ├── ZCR
     └── RMS
     ↓
Feature Fusion
     ↓
Conv1D
     ↓
Conv1D
     ↓
Bidirectional LSTM
     ↓
Temporal Attention
     ↓
Dense Layer
     ↓
Softmax
     ↓
Emotion Prediction




---

## Author

**Vandana Tanwar**  
B.Tech – Information Technology  
Chandigarh University

GitHub: [@thakurji12034-pixel](https://github.com/thakurji12034-pixel)

---

## License

This project is developed for educational and research purposes.
