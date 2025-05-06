import os
import torch

# === Paths ===
mel_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\preprocessed_tensors\mels"
label_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\preprocessed_tensors\labels"
print("Starting mismatch check...")
print(os.listdir(mel_dir)[:5])
# === Stats ===
mismatch_count = 0
total_count = 0
bad_files = []

for fname in os.listdir(mel_dir):
    if not fname.endswith("_mel.pt"):
        continue

    base = fname.replace("_mel.pt", "")
    mel_path = os.path.join(mel_dir, fname)
    label_path = os.path.join(label_dir, base + "_label.pt")

    if not os.path.exists(label_path):
        print(f"[MISSING LABEL] {base}")
        continue

    try:
        mel = torch.load(mel_path)
        label = torch.load(label_path)

        if mel.shape[0] != label.shape[0]:
            mismatch_count += 1
            bad_files.append(f"{base}: mel={mel.shape[0]}, label={label.shape[0]}")

        total_count += 1
    except Exception as e:
        print(f"[ERROR] Failed to load {base}: {e}")

print("\n=== MISMATCH SUMMARY ===")
print(f"Total pairs checked: {total_count}")
print(f"Frame mismatches   : {mismatch_count}")
if bad_files:
    print("\n--- Mismatched Files ---")
    for f in bad_files[:30]:
        print(f)
    if len(bad_files) > 30:
        print("... (truncated)")

