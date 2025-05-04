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

from model.model import CRNN
from trueDataset import PianoMAPSDataset

def compute_frame_metrics(preds, targets, threshold=0.1):
    preds = torch.sigmoid(preds)
    preds_bin = (preds > threshold).cpu().numpy().astype(int)
    targets_bin = (targets > 0.5).cpu().numpy().astype(int)

    precision_list, recall_list, f1_list, acc_list = [], [], [], []
    for p, t in zip(preds_bin, targets_bin):
        p_flat = p.flatten()
        t_flat = t.flatten()
        prec, rec, f1, _ = precision_recall_fscore_support(t_flat, p_flat, average='binary', zero_division=0)
        acc = accuracy_score(t_flat, p_flat)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        acc_list.append(acc)

    return {
        "frame_precision": sum(precision_list)/len(precision_list),
        "frame_recall": sum(recall_list)/len(recall_list),
        "frame_f1": sum(f1_list)/len(f1_list),
        "frame_accuracy": sum(acc_list)/len(acc_list)
    }


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
            notes.append((onset.item(), offset.item(), pitch))

    if len(notes) == 0:
        return np.zeros((0, 2)), np.zeros((0,))

    intervals = np.array([[on / fps, off / fps] for on, off, _ in notes])
    pitches = np.array([p for _, _, p in notes])
    return intervals, pitches


def compute_note_metrics(pred_roll, target_roll):
    ref_intervals, ref_pitches = extract_notes(target_roll)
    est_intervals, est_pitches = extract_notes(pred_roll)

    if len(ref_intervals) == 0 and len(est_intervals) == 0:
        return {"note_onset_f1": 1.0, "note_offset_f1": 1.0, "matched_onsets": 0, "matched_offsets": 0}

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


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./weights')
parser.add_argument('--save_every', type=int, default=5)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.save_dir)

train_loader = DataLoader(PianoMAPSDataset(args.data_dir, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(PianoMAPSDataset(args.data_dir, 'val'), batch_size=args.batch_size, shuffle=False, num_workers=2)

model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

csv_path = os.path.join(args.save_dir, 'loss_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'F1', 'Precision', 'Recall', 'Accuracy'])

best_val_loss = float('inf')

for epoch in range(1, args.num_epochs + 1):
    model.train()
    train_loss = 0.0
    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        frame_out, onset_out = model(mel)
        loss = criterion(frame_out, label) + 5.0 * criterion(onset_out, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            frame_out, onset_out = model(mel)
            loss = criterion(frame_out, label) + 5.0 * criterion(onset_out, label)
            val_loss += loss.item()
            all_preds.append(torch.sigmoid(frame_out))
            all_targets.append(label)

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_frame_metrics(preds, targets)

    note_preds = (preds > 0.1).float()
    note_targets = (targets > 0.5).float()

    note_metrics = [compute_note_metrics(p.cpu(), t.cpu()) for p, t in zip(note_preds, note_targets)]
    note_on_f1 = np.mean([m["note_onset_f1"] for m in note_metrics])
    note_off_f1 = np.mean([m["note_offset_f1"] for m in note_metrics])
    matched_on = sum([m["matched_onsets"] for m in note_metrics])
    matched_off = sum([m["matched_offsets"] for m in note_metrics])

    print(f"\n[Epoch {epoch}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Frame F1: {metrics['frame_f1']:.4f} | Precision: {metrics['frame_precision']:.4f} | Recall: {metrics['frame_recall']:.4f} | Acc: {metrics['frame_accuracy']:.4f}")
    print(f"Note Onset F1: {note_on_f1:.4f} | Note Offset F1: {note_off_f1:.4f} | Matched Onsets: {matched_on} | Matched Offsets: {matched_off}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
        print("New best model saved.")

    if epoch % args.save_every == 0:
        ckpt_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

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
