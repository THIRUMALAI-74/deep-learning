# ========================================================
# train_bird_model.py
# Train bird sound classifier and save model + label mapping
# ========================================================

import os, random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# ------------------- Config -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_SIZE = 128
SECONDS = 5.0
SR = 22050
MAX_FRAMES = IMG_SIZE
MIN_SAMPLES_PER_CLASS = 20
TOP_K = 30
MODEL_NAME = "bird_sound_resnet50.h5"
LABELS_CSV = "label_mapping.csv"

# ------------------- Utils -------------------
def audio_to_mel(path, sr=SR, seconds=SECONDS, n_mels=IMG_SIZE, max_frames=MAX_FRAMES):
    """Convert audio file -> mel spectrogram (3-channel)"""
    try:
        y, _ = librosa.load(path, sr=sr, duration=seconds)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Resize/pad to fixed size
        if mel_norm.shape[1] < max_frames:
            pad = max_frames - mel_norm.shape[1]
            mel_norm = np.pad(mel_norm, ((0,0),(0,pad)), mode="constant")
        else:
            mel_norm = mel_norm[:, :max_frames]

        mel_norm = mel_norm.astype(np.float32)

        # ✅ Convert (128,128,1) → (128,128,3) by repeating channel
        mel_3ch = np.repeat(mel_norm[..., np.newaxis], 3, axis=-1)

        return mel_3ch
    except Exception:
        return None

def build_model(num_classes):
    input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------- Training -------------------
def train_model(dataset_dir):
    # Step 1: Collect files
    audio_exts = (".wav",".mp3",".ogg",".flac",".m4a")
    rows = []
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith(audio_exts):
                full = os.path.join(root, f)
                label = os.path.basename(os.path.dirname(full))
                rows.append((full, label))
    files_df = pd.DataFrame(rows, columns=["path","label"])
    print(f"Total audio files: {len(files_df)}")

    # Step 2: Filter classes
    class_counts = files_df["label"].value_counts()
    eligible = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    chosen = class_counts.loc[eligible].nlargest(TOP_K).index.tolist() if TOP_K else eligible
    files_df = files_df[files_df["label"].isin(chosen)]
    print(f"Using {len(files_df)} files across {files_df['label'].nunique()} classes")

    # Step 3: Convert to spectrograms
    spectrograms, labels = [], []
    for p,lbl in tqdm(zip(files_df["path"], files_df["label"]), total=len(files_df)):
        mel = audio_to_mel(p)
        if mel is not None:
            spectrograms.append(mel)
            labels.append(lbl)

    # ✅ Already (128,128,3), no extra newaxis
    X = np.array(spectrograms)

    # Step 4: Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    y_cat = to_categorical(y_enc)
    num_classes = y_cat.shape[1]
    pd.DataFrame({"label": le.classes_, "index": range(len(le.classes_))}).to_csv(LABELS_CSV, index=False)
    print("Saved label mapping to", LABELS_CSV)

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y_cat, random_state=SEED)

    # Step 6: Build + Train
    model = build_model(num_classes)
    datagen = ImageDataGenerator(width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True)
    train_gen = datagen.flow(X_train, y_train, batch_size=16)
    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    model.fit(train_gen, validation_data=(X_test, y_test), epochs=30, callbacks=[es], verbose=1)

    # Step 7: Save
    model.save(MODEL_NAME)
    print("Model saved to", MODEL_NAME)

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {acc:.4f}")

# ------------------- Main -------------------
if __name__ == "__main__":
    dataset_path = "Voice of Birds"   # <-- change to your dataset folder
    train_model(dataset_path)
