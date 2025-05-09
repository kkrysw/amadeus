import os
import torch

# === Paths ===
mel_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\true_preprocessed_tensors\mels"
label_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\true_preprocessed_tensors\labels"
print("Starting mismatch check...")
print(os.listdir(mel_dir)[:5])
# === Stats ===
mismatch_count = 0
total_count = 0
bad_files = []

sparse_bins = {"<1": 0, "<5": 0, "<10": 0, "<20": 0, "<50": 0, ">=50": 0}


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
        else:
            nonzero = label.sum().item()
            if nonzero < 1:
                sparse_bins["<1"] += 1
            elif nonzero < 5:
                sparse_bins["<5"] += 1
            elif nonzero < 10:
                sparse_bins["<10"] += 1
            elif nonzero < 20:
                sparse_bins["<20"] += 1
            elif nonzero < 50:
                sparse_bins["<50"] += 1
            else:
                sparse_bins[">=50"] += 1

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

print("\n=== LABEL SPARSITY DISTRIBUTION ===")
for k, v in sparse_bins.items():
    print(f"{k} non-zeros: {v}")
