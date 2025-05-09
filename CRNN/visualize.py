import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# --- PATHS ---
BASE_DIR = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\true_preprocessed_tensors"
MEL_DIR = os.path.join(BASE_DIR, "mels")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

# --- CATEGORY MAP ---
group_data = defaultdict(lambda: {'mel': [], 'label': [], 'onset': []})

def safe_group(filename):
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[1]  # e.g., "ISOL", "UCHO", etc.
    return "UNKNOWN"


# --- COLLECT ---
print("[INFO] Scanning all tensors and grouping...")
for fname in tqdm(sorted(os.listdir(MEL_DIR))):
    if not fname.endswith("_mel.pt"):
        continue

    base = fname.replace("_mel.pt", "")
    group = safe_group(base)

    mel_path = os.path.join(MEL_DIR, fname)
    label_path = os.path.join(LABEL_DIR, f"{base}_label.pt")

    if not os.path.exists(label_path):
        continue

    mel = torch.load(mel_path).float().T  # [mel_bins, T]
    label = torch.load(label_path).float().T  # [88, T]

    onset = torch.zeros_like(label)
    onset[:, 1:] = (label[:, 1:] > 0.5) & (label[:, :-1] == 0)

    T = min(mel.shape[1], label.shape[1])
    mel = mel[:, :T]
    label = label[:, :T]
    onset = onset[:, :T]

    group_data[group]['mel'].append(mel)
    group_data[group]['label'].append(label)
    group_data[group]['onset'].append(onset)

# --- PLOT PER GROUP ---
for group, tensors in group_data.items():
    mels = tensors['mel']
    labels = tensors['label']
    onsets = tensors['onset']

    if len(mels) == 0:
        continue

    min_T = min(mel.shape[1] for mel in mels)
    mel_sum = sum([m[:, :min_T] for m in mels])
    label_sum = sum([l[:, :min_T] for l in labels])
    onset_sum = sum([o[:, :min_T] for o in onsets])

    count = len(mels)
    mel_avg = mel_sum / count
    label_avg = label_sum / count
    onset_avg = onset_sum / count

    print(f"\n[GROUP: {group}] Count: {count}")
    print(f"Avg label activity: {label_avg.mean().item():.6f}, Avg onset activity: {onset_avg.mean().item():.6f}")

    fig, axs = plt.subplots(3, 1, figsize=(14, 9), dpi=120)
    axs[0].imshow(mel_avg.numpy(), aspect='auto', origin='lower', cmap='magma')
    axs[0].set_title(f"{group} - Avg Mel Spectrogram")
    axs[0].set_ylabel("Mel Bins")

    axs[1].imshow(label_avg.numpy(), aspect='auto', origin='lower', cmap='Greens')
    axs[1].set_title(f"{group} - Avg Note Activation")
    axs[1].set_ylabel("MIDI Pitch")

    axs[2].imshow(onset_avg.numpy(), aspect='auto', origin='lower', cmap='Reds')
    axs[2].set_title(f"{group} - Avg Note Onsets (Inferred)")
    axs[2].set_ylabel("MIDI Pitch")
    axs[2].set_xlabel("Time Frames")

    plt.tight_layout()
    plt.savefig(f"{group}_avg_tensor_summary.png")
    plt.close()

print("\n[✓] All group-wise plots saved.")
