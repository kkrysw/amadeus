import os
import torchaudio
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram

# Adjust these paths for local use
audio_dir = "/Users/krystalwu/ReconVAT/MAPS/newMAP/audio"
output_dir = "/Users/krystalwu/ReconVAT/MAPS/preprocessed_tensors"

mel_transform = MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=229)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "mels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

def safe_bounds(val, max_val):
    return max(0, min(val, max_val))

for fname in tqdm(os.listdir(audio_dir)):
    if not fname.endswith(".flac"):
        continue

    base = os.path.splitext(fname)[0]
    flac_path = os.path.join(audio_dir, base + ".flac")
    tsv_path = os.path.join(audio_dir, base + ".tsv")

    if not os.path.exists(tsv_path):
        print(f"[SKIP] Missing TSV for {base}")
        continue

    try:
        wav, sr = torchaudio.load(flac_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        mel = mel_transform(wav).squeeze(0).T  # Shape: (T, 229)
        n_frames = mel.shape[0]

        df = pd.read_csv(tsv_path, sep="\t", names=["onset", "offset", "pitch", "velocity"])
        df["pitch"] = pd.to_numeric(df["pitch"], errors="coerce")
        df = df.dropna(subset=["pitch"])
        df = df[df["pitch"] > 0]

        label = np.zeros((n_frames, 88), dtype=np.float32)

        for _, row in df.iterrows():
            on = int(float(row["onset"]) * 100)
            off = int(float(row["offset"]) * 100)
            pitch = int(row["pitch"]) - 21
            if 0 <= pitch < 88 and off > on:
                on = safe_bounds(on, n_frames - 1)
                off = safe_bounds(off, n_frames)
                if off > on:
                    label[on:off, pitch] = 1.0

        # Save tensors with expected suffixes
        torch.save(mel, os.path.join(output_dir, "mels", f"{base}_mel.pt"))
        torch.save(torch.tensor(label), os.path.join(output_dir, "labels", f"{base}_label.pt"))

    except Exception as e:
        print(f"[ERROR] {base}: {e}")
