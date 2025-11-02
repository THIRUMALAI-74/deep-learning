# ========================================================
# predict_bird.py
# Predict bird species from a given audio file
# ========================================================

import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model

# ------------------- Config -------------------
MODEL_NAME = r"D:\Projects\Bird sound detector\ResNet50\bird_sound_resnet50.h5"
LABELS_CSV = r"D:\Projects\Bird sound detector\ResNet50\label_mapping.csv"
IMG_SIZE = 128
SECONDS = 5.0
SR = 22050
MAX_FRAMES = IMG_SIZE

# ------------------- Utils -------------------
def audio_to_mel(path, sr=SR, seconds=SECONDS, n_mels=IMG_SIZE, max_frames=MAX_FRAMES):
    """Convert audio file -> mel spectrogram (3-channel)"""
    try:
        y, _ = librosa.load(path, sr=sr, duration=seconds)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        if mel_norm.shape[1] < max_frames:
            pad = max_frames - mel_norm.shape[1]
            mel_norm = np.pad(mel_norm, ((0,0),(0,pad)), mode="constant")
        else:
            mel_norm = mel_norm[:, :max_frames]

        # Convert to 3-channel
        mel_3ch = np.repeat(mel_norm[..., np.newaxis], 3, axis=-1)
        return np.array([mel_3ch], dtype=np.float32)   # batch dimension
    except Exception as e:
        print("Error loading audio:", e)
        return None

# ------------------- Load model + labels -------------------
model = load_model(MODEL_NAME)
le_df = pd.read_csv(LABELS_CSV)
labels = le_df['label'].tolist()

# ------------------- Prediction function -------------------
def predict_bird(audio_path):
    X = audio_to_mel(audio_path)
    if X is None:
        return "Could not process audio"
    
    pred = model.predict(X)
    pred_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][pred_idx]
    bird_name = labels[pred_idx]
    print("\n==================== USING RESNET50 MODEL ====================\n")
    print(f"Predicted Bird: {bird_name}  |  Confidence: {confidence*100:.2f}%")
    return bird_name, confidence

# ------------------- Main -------------------
if __name__ == "__main__":
    audio_file =r"D:\Projects\Bird sound detector\ResNet50\Guan6.mp3"
    predict_bird(audio_file)
