
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mir_eval

from model.model import CRNN
from localTrueDataset import LocalPianoMAPSDataset


def collate_pad_fn(batch, global_len=408):
    import torch.nn.functional as F
    mels, labels, onsets = zip(*batch)

    padded_mels = [F.pad(mel, (0, global_len - mel.shape[-1])) for mel in mels]  # keep [1, 229, T]
    padded_labels = [F.pad(label, (0, 0, 0, global_len - label.shape[0])) for label in labels]  # [T, 88]
    padded_onsets = [F.pad(onset, (0, 0, 0, global_len - onset.shape[0])) for onset in onsets]  # [T, 88]

    return (
        torch.stack(padded_mels),                            # → [B, 1, 229, T]
        torch.stack(padded_labels),                          # → [B, T, 88]
        torch.stack(padded_onsets),                          # → [B, T, 88]
    )


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_dir = "/content/preprocessed_tensors"
save_dir = "/content/weights_local"
os.makedirs(save_dir, exist_ok=True)

model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(DEVICE))

train_loader = DataLoader(
    LocalPianoMAPSDataset(tensor_dir, 'train'),
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_pad_fn
)

val_loader = DataLoader(
    LocalPianoMAPSDataset(tensor_dir, 'val'),
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_pad_fn
)

csv_path = os.path.join(save_dir, 'loss_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'F1', 'Precision', 'Recall', 'Accuracy'])

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for mel, label, onset in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
        mel, label, onset = mel.to(DEVICE), label.to(DEVICE), onset.to(DEVICE)
        frame_out, onset_out = model(mel)
        loss_frame = criterion(frame_out, label)
        loss_onset = criterion(onset_out, onset)
        loss = 0.5 * loss_frame + 0.5 * loss_onset

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    model.eval()
    val_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for batch_idx, (mel, label, onset) in enumerate(val_loader):
            mel, label, onset = mel.to(DEVICE), label.to(DEVICE), onset.to(DEVICE)
            frame_out, onset_out = model(mel)
            loss = 0.5 * criterion(frame_out, label) + 0.5 * criterion(onset_out, onset)
            val_loss += loss.item()

            sigmoid_out = torch.sigmoid(frame_out).cpu()
            all_preds.append(sigmoid_out)
            all_targets.append(label.cpu())

            if epoch == 1 and batch_idx == 0:
                heat = sigmoid_out[0].numpy().T
                plt.figure(figsize=(10, 5))
                plt.imshow(heat, aspect='auto', origin='lower', cmap='magma')
                plt.title(f"Sigmoid Heatmap Epoch {epoch}")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"{save_dir}/heatmap_epoch{epoch}.png")
                plt.close()

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    best_f1 = 0
    best_thresh = 0

    for thresh in [0.005, 0.01, 0.05, 0.1, 0.2]:
        pred_bin = (preds > thresh).numpy().astype(int)
        target_bin = (targets > 0.5).numpy().astype(int)

        tn, fp, fn, tp = confusion_matrix(target_bin.flatten(), pred_bin.flatten(), labels=[0, 1]).ravel()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"[Epoch {epoch}] Best Threshold: {best_thresh:.3f} with F1: {best_f1:.4f}")

    # === mir_eval metrics ===
    ref_intervals, ref_pitches = [], []
    est_intervals, est_pitches = [], []
    for b in range(targets.shape[0]):
        for t in range(targets.shape[1]):
            for p in range(targets.shape[2]):
                if targets[b, t, p] > 0.5:
                    ref_intervals.append([t / 100.0, (t + 1) / 100.0])
                    ref_pitches.append(p + 21)
                if preds[b, t, p] > best_thresh:
                    est_intervals.append([t / 100.0, (t + 1) / 100.0])
                    est_pitches.append(p + 21)
    if ref_intervals and est_intervals:
        mir_p, mir_r, mir_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            np.array(ref_intervals), np.array(ref_pitches),
            np.array(est_intervals), np.array(est_pitches), onset_tolerance=0.05
        )
        print(f"[mir_eval] Precision: {mir_p:.4f}, Recall: {mir_r:.4f}, F1: {mir_f:.4f}")
    else:
        print("[mir_eval] No valid note events detected.")

    scheduler.step()
    print("Training finished.")

