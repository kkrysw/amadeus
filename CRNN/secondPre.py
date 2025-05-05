import os
import torchaudio
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram

# Adjust these
audio_dir = "/scratch/ksw9582/MAPS/newMAP/audio"
output_dir = "/scratch/ksw9582/MAPS/preprocessed_tensors"

mel_transform = MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=229)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "mels"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

def split(filename):
    return ['train', 'val', 'test'][hash(filename) % 3]

for fname in tqdm(os.listdir(audio_dir)):
    if fname.endswith(".flac"):
        base = os.path.splitext(fname)[0]
        flac_path = os.path.join(audio_dir, base + ".flac")
        tsv_path = os.path.join(audio_dir, base + ".tsv")

        if not os.path.exists(tsv_path):
            print(f"Missing TSV for {base}")
            continue

        try:
            wav, sr = torchaudio.load(flac_path)
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)

            mel = mel_transform(wav).squeeze(0).T  # (T, 229)
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
                    on = min(max(on, 0), n_frames - 1)
                    off = min(max(off, on + 1), n_frames)
                    label[on:off, pitch] = 1.0

            torch.save(mel, os.path.join(output_dir, "mels", f"{base}_mel.pt"))
            torch.save(torch.tensor(label), os.path.join(output_dir, "labels", f"{base}_label.pt"))

        except Exception as e:
            print(f"Error processing {base}: {e}")
