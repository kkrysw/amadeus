import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
from model.model import CRNN
from localTrueDataset import LocalPianoMAPSDataset
from sklearn.metrics import confusion_matrix
import mir_eval

def piano_roll_to_notes(piano_roll, frame_hop_s=0.032, midi_offset=21):
    """Convert [T, 88] piano roll to note onset list."""
    import numpy as np
    T, pitch_dim = piano_roll.shape
    onsets = []
    pitches = []
    for pitch_idx in range(pitch_dim):
        active = np.where(piano_roll[:, pitch_idx] > 0)[0]
        if len(active) > 0:
            # group by consecutive frames
            groups = np.split(active, np.where(np.diff(active) > 1)[0] + 1)
            for group in groups:
                onset_frame = group[0]
                onsets.append(onset_frame * frame_hop_s)
                pitches.append(pitch_idx + midi_offset)
    return np.array(onsets), np.array(pitches)

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

# Set local tensor path
tensor_dir = "/content/preprocessed_tensors"
save_dir = "/content/weights_local"
os.makedirs(save_dir, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

if __name__ == "__main__":
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
        csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'Framewsie F1', 'Framewise Precision', 'Framewise Recall', 'Framewise Accuracy', 'Notewise F1', 'Notewise Precision', "Notewise Recall"])

    import time
    start = time.time()
    print("STARTING!\n")

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for mel, label, onset in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            mel = mel.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            onset = onset.to(DEVICE, non_blocking=True)

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

        frame_hop_s = 0.032
        ref_onsets_all, ref_pitches_all = [], []
        est_onsets_all, est_pitches_all = [], []

        model.eval()
        val_loss=0
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        with torch.no_grad():
            for mel, label, onset in val_loader:
                mel, label, onset = mel.to(DEVICE), label.to(DEVICE), onset.to(DEVICE)
                frame_out, onset_out = model(mel)
                loss = 0.5 * criterion(frame_out, label) + 0.5 * criterion(onset_out, onset)
                val_loss += loss.item()
                pred=torch.sigmoid(frame_out).cpu()
                target=label.cpu()

                pred_bin = (pred > 0.001).numpy().astype(int)
                target_bin = (target > 0.5).numpy().astype(int)

                for b in range(pred_bin.shape[0]):
                    ref_onsets, ref_pitches = piano_roll_to_notes(target_bin[b], frame_hop_s)
                    est_onsets, est_pitches = piano_roll_to_notes(pred_bin[b], frame_hop_s)
                    ref_onsets_all.append(ref_onsets)
                    ref_pitches_all.append(ref_pitches)
                    est_onsets_all.append(est_onsets)
                    est_pitches_all.append(est_pitches)

                pred_flat = pred_bin.flatten()
                target_flat = target_bin.flatten()
            
                tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0,1]).ravel()
            
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

        avg_val_loss = val_loss / len(val_loader)

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-8)

        metrics = {
            'frame_precision': precision,
            'frame_recall': recall,
            'frame_f1': f1,
            'frame_accuracy': accuracy
        }

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Frame F1: {metrics['frame_f1']:.4f} | Precision: {metrics['frame_precision']:.4f} | Recall: {metrics['frame_recall']:.4f}")

        mir_precision, mir_recall, mir_f1 = 0, 0, 0
        for ref_onsets, ref_pitches, est_onsets, est_pitches in zip(
            ref_onsets_all, ref_pitches_all, est_onsets_all, est_pitches_all):
            p, r, f = mir_eval.transcription.onset_precision_recall_f1(
            ref_onsets, ref_pitches, est_onsets, est_pitches, onset_tolerance=0.05)
            mir_precision += p
            mir_recall += r
            mir_f1 += f

        mir_precision /= len(ref_onsets_all)
        mir_recall /= len(ref_onsets_all)
        mir_f1 /= len(ref_onsets_all)

        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, avg_train_loss, avg_val_loss,
                metrics['frame_f1'], metrics['frame_precision'],
                metrics['frame_recall'], metrics['frame_accuracy'],
                mir_f1, mir_precision, mir_recall
            ])

        scheduler.step()

    print("Local training finished.")