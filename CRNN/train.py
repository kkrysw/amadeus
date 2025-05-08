import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import mir_eval
import numpy as np
import matplotlib.pyplot as plt
import traceback

from model.model import CRNN
from trueDataset import PianoMAPSDataset

def compute_frame_metrics(preds, targets, threshold=0.1):
    preds_bin = (preds > threshold).cpu().numpy().astype(int)
    targets_bin = (targets > 0.5).cpu().numpy().astype(int)
    p, r, f1, _ = precision_recall_fscore_support(targets_bin.flatten(), preds_bin.flatten(), average='binary', zero_division=0)
    acc = accuracy_score(targets_bin.flatten(), preds_bin.flatten())
    return {"frame_precision": p, "frame_recall": r, "frame_f1": f1, "frame_accuracy": acc}

def collate_pad_fn(batch):
    import torch.nn.functional as F
    mels, labels, onsets = zip(*batch)
    max_len = max(mel.shape[-1] for mel in mels)

    padded_mels = [F.pad(mel, (0, max_len - mel.shape[-1])) for mel in mels]
    padded_labels = [F.pad(label, (0, 0, 0, max_len - label.shape[0])) for label in labels]
    padded_onsets = [F.pad(onset, (0, 0, 0, max_len - onset.shape[0])) for onset in onsets]

    return torch.stack(padded_mels), torch.stack(padded_labels), torch.stack(padded_onsets)

def extract_notes(piano_roll, onset_thresh=0.5, fps=100):
    notes = []
    for pitch in range(piano_roll.shape[1]):
        active = piano_roll[:, pitch] > onset_thresh
        changes = torch.diff(active.int())
        onsets = torch.where(changes == 1)[0]
        offsets = torch.where(changes == -1)[0]
        if len(onsets) > 0 and (len(offsets) == 0 or offsets[0] < onsets[0]):
            offsets = torch.cat([offsets, torch.tensor([piano_roll.shape[0]-1])])
        if len(onsets) > len(offsets):
            onsets = onsets[:len(offsets)]
        for onset, offset in zip(onsets, offsets):
            if offset > onset:
                notes.append((onset.item(), offset.item(), pitch))
    if len(notes) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    intervals = np.array([[on / fps, off / fps] for on, off, _ in notes])
    pitches = np.array([p for _, _, p in notes])
    return intervals, pitches

def compute_note_metrics(pred_roll, target_roll):
    ref_intervals, ref_pitches = extract_notes(target_roll)
    est_intervals, est_pitches = extract_notes(pred_roll)

    if len(ref_intervals) == 0 or len(est_intervals) == 0:
        return None  # Skip this pair

    if np.any(ref_pitches <= 0) or np.any(est_pitches <= 0):
        return None  # Skip invalid pitches

    try:
        _, _, f_on, matched_on = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=None)
        _, _, f_off, matched_off = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches, offset_ratio=0.2)
        return {
            "note_onset_f1": f_on,
            "note_offset_f1": f_off,
            "matched_onsets": matched_on,
            "matched_offsets": matched_off
        }
    except Exception as e:
        print(f"[WARN] mir_eval failed: {e}")
        return None

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./weights')
parser.add_argument('--save_every', type=int, default=5)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--note_eval_subset', type=int, default=64)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(log_dir=args.save_dir)
os.makedirs(args.save_dir, exist_ok=True)

model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

train_loader = DataLoader(
    PianoMAPSDataset(args.data_dir, 'train'),
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_pad_fn
)

val_loader = DataLoader(
    PianoMAPSDataset(args.data_dir, 'val'),
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate_pad_fn
)

csv_path = os.path.join(args.save_dir, 'loss_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'F1', 'Precision', 'Recall', 'Accuracy'])

best_val_loss = float('inf')

for epoch in range(1, args.num_epochs + 1):
    model.train()
    train_loss = 0.0
    try:
        for batch_idx, (mel, label, onset) in enumerate(train_loader):
            mel, label, onset = mel.to(DEVICE), label.to(DEVICE), onset.to(DEVICE)
            frame_out, onset_out = model(mel)
            loss = criterion(frame_out, label) + 0.5 * criterion(onset_out, onset)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            if args.debug and batch_idx == 0:
                pred = torch.sigmoid(frame_out[0]).detach().cpu().numpy()
                gt = label[0].cpu().numpy()
                plt.figure(figsize=(10, 3))
                plt.subplot(1, 2, 1); plt.imshow(pred.T, aspect='auto'); plt.title('Predicted')
                plt.subplot(1, 2, 2); plt.imshow(gt.T, aspect='auto'); plt.title('Ground Truth')
                plt.savefig(f"{args.save_dir}/epoch_{epoch}_sample0.png")
                plt.close()
    except Exception:
        traceback.print_exc()
        break

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for mel, label, onset in tqdm(val_loader):
            mel, label, onset = mel.to(DEVICE), label.to(DEVICE), onset.to(DEVICE)
            frame_out, onset_out = model(mel)
            loss = criterion(frame_out, label) + 0.5 * criterion(onset_out, onset)
            val_loss += loss.item()
            all_preds.append(torch.sigmoid(frame_out).cpu())
            all_targets.append(label.cpu())

    avg_val_loss = val_loss / len(val_loader)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    
    metrics = compute_frame_metrics(preds, targets)

    note_preds = (preds > 0.1).float()
    note_targets = (targets > 0.5).float()
    eval_subset = min(args.note_eval_subset, note_preds.shape[0])

    note_metrics_raw = [compute_note_metrics(p.cpu(), t.cpu()) for p, t in zip(note_preds, note_targets)]
    note_metrics = [m for m in note_metrics_raw if m is not None]

    if len(note_metrics) == 0:
        print("[WARN] No valid note metrics this epoch.")
        note_on_f1 = note_off_f1 = matched_on = matched_off = 0.0
    else:
        note_on_f1 = np.mean([m["note_onset_f1"] for m in note_metrics])
        note_off_f1 = np.mean([m["note_offset_f1"] for m in note_metrics])
        matched_on = sum([m["matched_onsets"] for m in note_metrics])
        matched_off = sum([m["matched_offsets"] for m in note_metrics])


    #note_on_f1 = np.mean([m["note_onset_f1"] for m in note_metrics])
    #note_off_f1 = np.mean([m["note_offset_f1"] for m in note_metrics])

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Frame F1: {metrics['frame_f1']:.4f} | Note Onset F1: {note_on_f1:.4f} | Note Offset F1: {note_off_f1:.4f}")
    skipped_count = len(note_metrics_raw) - len(note_metrics)
    print(f"[INFO] Skipped {skipped_count} invalid note pairs in epoch {epoch}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
        print("New best model saved.")

    if epoch % args.save_every == 0:
        ckpt_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)
    onset_preds = torch.sigmoid(onset_out)
    onset_metrics = compute_frame_metrics(onset_preds.cpu(), onset.cpu())
    print(f"Onset F1: {onset_metrics['frame_f1']:.4f} | Onset Precision: {onset_metrics['frame_precision']:.4f} | Onset Recall: {onset_metrics['frame_recall']:.4f}")

    writer.add_scalar('Metrics/Onset_F1', onset_metrics['frame_f1'], epoch)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Val', avg_val_loss, epoch)
    writer.add_scalar('Metrics/F1', metrics['frame_f1'], epoch)
    writer.add_scalar('Metrics/Precision', metrics['frame_precision'], epoch)
    writer.add_scalar('Metrics/Recall', metrics['frame_recall'], epoch)
    writer.add_scalar('Metrics/Accuracy', metrics['frame_accuracy'], epoch)

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch, avg_train_loss, avg_val_loss,
            metrics['frame_f1'], metrics['frame_precision'],
            metrics['frame_recall'], metrics['frame_accuracy']
        ])

print("Training complete.")
writer.close()
