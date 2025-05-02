import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
import mir_eval
import numpy as np
import pickle
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

def extract_notes(piano_roll, fs=100):
    notes = []
    piano_roll = piano_roll.cpu().numpy()
    for pitch in range(piano_roll.shape[1]):
        active = piano_roll[:, pitch] > 0.5
        changes = np.diff(active.astype(int), prepend=0)
        onsets = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]
        if len(offsets) < len(onsets):
            offsets = np.append(offsets, piano_roll.shape[0] - 1)
        for on, off in zip(onsets, offsets):
            onset_time = on / fs
            offset_time = off / fs
            midi_pitch = pitch + 21
            notes.append(((onset_time, offset_time), midi_pitch))
    return notes

def compute_note_metrics(pred_roll, target_roll, fs=100):
    pred_notes = extract_notes(pred_roll, fs)
    target_notes = extract_notes(target_roll, fs)
    if not pred_notes or not target_notes:
        return {
            "note_onset_F1": 0.0,
            "note_offset_F1": 0.0,
            "matched_onset": 0,
            "matched_offset": 0,
            "num_pred": len(pred_notes),
            "num_target": len(target_notes)
        }
    ref_int, ref_pitch = zip(*target_notes)
    est_int, est_pitch = zip(*pred_notes)
    _, _, f_on, matched_on = mir_eval.transcription.precision_recall_f1_overlap(ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None)
    _, _, f_off, matched_off = mir_eval.transcription.precision_recall_f1_overlap(ref_int, ref_pitch, est_int, est_pitch, offset_ratio=0.2)
    return {
        "note_onset_F1": f_on,
        "note_offset_F1": f_off,
        "matched_onset": matched_on,
        "matched_offset": matched_off,
        "num_pred": len(pred_notes),
        "num_target": len(target_notes)
    }

parser = argparse.ArgumentParser(description="Train CRNN on MAPS Dataset")
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

train_dataset = PianoMAPSDataset(args.data_dir, split='train')
val_dataset = PianoMAPSDataset(args.data_dir, split='val')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
        print("Label max:", label.max().item(), "min:", label.min().item(), "mean:", label.mean().item())
        frame_out, onset_out = model(mel)
        optimizer.zero_grad()
        frame_loss = criterion(frame_out, (label > 0).float())
        onset_loss = criterion(onset_out, (label > 0).float())
        loss = frame_loss + 5.0 * onset_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            frame_out, onset_out = model(mel)
            frame_loss = criterion(frame_out, (label > 0).float())
            onset_loss = criterion(onset_out, (label > 0).float())
            val_loss += (frame_loss + 5.0 * onset_loss).item()
            all_preds.append(torch.sigmoid(frame_out))
            all_targets.append(label)

            if epoch == 1:
                pred_plot = torch.sigmoid(frame_out[0]).detach().cpu().numpy()
                label_plot = label[0].detach().cpu().numpy()
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(pred_plot.T, aspect='auto', origin='lower')
                plt.title('Predicted')
                plt.subplot(1, 2, 2)
                plt.imshow(label_plot.T, aspect='auto', origin='lower')
                plt.title('Label')
                plt.savefig(f"{args.save_dir}/epoch_{epoch}_vis.png")
                plt.close()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    print(f"Pred logits: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}")
    metrics = compute_frame_metrics(preds, targets)

    preds_bin = (preds > 0.1).float()
    targets_bin = (targets > 0.5).float()

    onset_f1s, offset_f1s = [], []
    matched_on, matched_off, total_pred, total_tgt = [], [], [], []

    for pred, target in zip(preds_bin, targets_bin):
        m = compute_note_metrics(pred, target)
        onset_f1s.append(m['note_onset_F1'])
        offset_f1s.append(m['note_offset_F1'])
        matched_on.append(m['matched_onset'])
        matched_off.append(m['matched_offset'])
        total_pred.append(m['num_pred'])
        total_tgt.append(m['num_target'])

    mean_onset_f1 = np.mean(onset_f1s)
    mean_offset_f1 = np.mean(offset_f1s)
    std_onset_f1 = np.std(onset_f1s)
    std_offset_f1 = np.std(offset_f1s)

    print(f"Matched Onsets     : {sum(matched_on)} / {sum(total_tgt)}")
    print(f"Matched Offsets    : {sum(matched_off)} / {sum(total_tgt)}")
    print(f"Note Onset F1      : {mean_onset_f1*100:.2f} ± {std_onset_f1*100:.2f}")
    print(f"Note Offset F1     : {mean_offset_f1*100:.2f} ± {std_offset_f1*100:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
        print(f"New best model saved at epoch {epoch}")

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
            epoch,
            avg_train_loss,
            avg_val_loss,
            metrics['frame_f1'],
            metrics['frame_precision'],
            metrics['frame_recall'],
            metrics['frame_accuracy']
        ])

print("Training complete.")
writer.close()
