import os
import torchaudio
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram

# === Settings ===
audio_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\newMAP\audio"
output_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\real_preprocessed_batches"
TARGET_FRAMES = 1024
MEL_BANDS = 128
BATCH_SIZE = 1000

mel_transform = MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=MEL_BANDS)

os.makedirs(output_dir, exist_ok=True)

def safe_bounds(val, max_val):
    return max(0, min(val, max_val))

mel_batch = []
label_batch = []
batch_count = 0
processed = 0
skipped = 0

for fname in tqdm(sorted(os.listdir(audio_dir))):
    if not fname.endswith(".flac"):
        continue

    base = os.path.splitext(fname)[0]
    flac_path = os.path.join(audio_dir, base + ".flac")
    tsv_path = os.path.join(audio_dir, base + ".tsv")

    if not os.path.exists(tsv_path):
        print(f"[SKIPPED] {base} - Missing TSV")
        skipped += 1
        continue

    try:
        wav, sr = torchaudio.load(flac_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        mel = mel_transform(wav).squeeze(0).T  # [T, 128]
        n_frames = mel.shape[0]
        mel = mel[:TARGET_FRAMES] if n_frames >= TARGET_FRAMES else torch.cat(
            [mel, torch.zeros(TARGET_FRAMES - n_frames, MEL_BANDS)], dim=0)

        label = np.zeros((n_frames, 88), dtype=np.float32)
        df = pd.read_csv(tsv_path, sep="\t", names=["onset", "offset", "pitch", "velocity"])
        df["pitch"] = pd.to_numeric(df["pitch"], errors="coerce")
        df = df.dropna(subset=["pitch"])
        df = df[df["pitch"] > 0]

        for _, row in df.iterrows():
            on = int(float(row["onset"]) * 100)
            off = int(float(row["offset"]) * 100)
            pitch = int(row["pitch"]) - 21
            if 0 <= pitch < 88 and off > on:
                on = safe_bounds(on, n_frames - 1)
                off = safe_bounds(off, n_frames)
                if off > on:
                    label[on:off, pitch] = 1.0

        label_tensor = torch.tensor(label)
        label_tensor = label_tensor[:TARGET_FRAMES] if n_frames >= TARGET_FRAMES else torch.cat(
            [label_tensor, torch.zeros(TARGET_FRAMES - n_frames, 88)], dim=0)

        mel_batch.append(mel.half())
        label_batch.append(label_tensor.half())
        processed += 1

        if len(mel_batch) == BATCH_SIZE:
            mel_tensor = torch.stack(mel_batch)
            label_tensor = torch.stack(label_batch)
            torch.save(mel_tensor, os.path.join(output_dir, f"mel_batch{batch_count}.pt"))
            torch.save(label_tensor, os.path.join(output_dir, f"label_batch{batch_count}.pt"))
            #print(f"[SAVED] Batch {batch_count}: {mel_tensor.shape}, {label_tensor.shape}")
            batch_count += 1
            mel_batch.clear()
            label_batch.clear()

    except Exception as e:
        print(f"[ERROR] {base}: {e}")
        skipped += 1

# Save remaining
if mel_batch:
    mel_tensor = torch.stack(mel_batch)
    label_tensor = torch.stack(label_batch)
    torch.save(mel_tensor, os.path.join(output_dir, f"mel_batch{batch_count}.pt"))
    torch.save(label_tensor, os.path.join(output_dir, f"label_batch{batch_count}.pt"))
    #print(f"[SAVED] Final Batch {batch_count}: {mel_tensor.shape}, {label_tensor.shape}")

print(f"\n[INFO] Done. {processed} files processed, {skipped} skipped, {batch_count+1} batch files saved.")
