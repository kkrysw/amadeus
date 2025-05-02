import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

from model.model import CRNN
from trueDataset import PianoMAPSDataset


def compute_frame_metrics(preds, targets, threshold=0.5):
    # Apply sigmoid + thresholding
    preds_bin = (preds > threshold).cpu().numpy().astype(int)
    targets_bin = (targets > 0.5).cpu().numpy().astype(int)

    # Flatten to compute metrics over all frames and notes
    p_flat = preds_bin.reshape(-1, 88)
    t_flat = targets_bin.reshape(-1, 88)

    # Use micro-average (preferred for multi-label binary)
    precision = precision_score(t_flat, p_flat, average='micro', zero_division=0)
    recall = recall_score(t_flat, p_flat, average='micro', zero_division=0)
    f1 = f1_score(t_flat, p_flat, average='micro', zero_division=0)
    acc = accuracy_score(t_flat, p_flat)

    return {
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
        "frame_accuracy": acc
    }

def extract_notes(piano_roll, onset_thresh=0.5, offset_thresh=0.5):
    notes = []
    for pitch in range(piano_roll.shape[1]):  # 88 notes
        active = piano_roll[:, pitch] > onset_thresh
        changes = active[:-1] != active[1:]
        indices = torch.where(changes)[0].cpu().numpy()
        if len(indices) % 2 != 0:
            indices = indices[:-1]  # ensure complete onset-offset pairs
        for i in range(0, len(indices), 2):
            onset = indices[i]
            offset = indices[i+1] if i+1 < len(indices) else onset + 1
            notes.append((onset, offset, pitch))
    return notes

def match_notes(pred_notes, target_notes, onset_tol=2):
    matched = 0
    used = set()
    for pred in pred_notes:
        for i, tgt in enumerate(target_notes):
            if i in used:
                continue
            # Compare onset within tolerance and same pitch
            if abs(pred[0] - tgt[0]) <= onset_tol and pred[2] == tgt[2]:
                matched += 1
                used.add(i)
                break
    return matched

def compute_note_metrics(pred_roll, target_roll, tol_frames=2):
    pred_notes = extract_notes(pred_roll)
    target_notes = extract_notes(target_roll)
    matched = match_notes(pred_notes, target_notes, onset_tol=tol_frames)

    P = matched / max(len(pred_notes), 1)
    R = matched / max(len(target_notes), 1)
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"note_precision": P, "note_recall": R, "note_f1": F1}


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train CRNN on MAPS Dataset")
parser.add_argument('--data_dir', type=str, required=True, help='Path to audio + tsv files')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./weights')
parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
args = parser.parse_args()

# --- Config ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.save_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.save_dir)

# --- Load Data ---
train_dataset = PianoMAPSDataset(args.data_dir, split='train')
val_dataset = PianoMAPSDataset(args.data_dir, split='val')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# --- Model + Optimizer ---
model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

# --- CSV Logging ---
csv_path = os.path.join(args.save_dir, 'loss_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss',
                             'F1', 'Precision', 'Recall', 'Accuracy'])

best_val_loss = float('inf')

# --- Training Loop ---
for epoch in range(1, args.num_epochs + 1):
    model.train()
    train_loss = 0.0

    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)  # [B, 229, 512], [B, 512, 88]
        frame_out, onset_out = model(mel)              # [B, time, 88]

        optimizer.zero_grad()
        frame_loss = criterion(frame_out, label.clamp(0, 1))
        onset_loss = criterion(onset_out, (label > 0).float())
        loss = frame_loss + onset_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            frame_out, onset_out = model(mel)

            frame_loss = criterion(frame_out, label.clamp(0, 1))
            onset_loss = criterion(onset_out, (label > 0).float())
            val_loss += (frame_loss + onset_loss).item()

            all_preds.append(torch.sigmoid(frame_out))
            all_targets.append(label)

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = compute_frame_metrics(preds, targets)

    # Note-based metrics (onset only, full note)
    note_preds = (preds > 0.5).float()
    note_targets = (targets > 0.5).float()

    note_metrics_list = [
        compute_note_metrics(pred.cpu(), tgt.cpu())
        for pred, tgt in zip(note_preds, note_targets)
    ]

    note_prec = sum([m['note_precision'] for m in note_metrics_list]) / len(note_metrics_list)
    note_rec  = sum([m['note_recall'] for m in note_metrics_list]) / len(note_metrics_list)
    note_f1   = sum([m['note_f1'] for m in note_metrics_list]) / len(note_metrics_list)

    metrics.update({
        "note_precision": note_prec,
        "note_recall": note_rec,
        "note_f1": note_f1
    })


    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Frame F1: {metrics['frame_f1']:.4f} | Precision: {metrics['frame_precision']:.4f} "
      f"| Recall: {metrics['frame_recall']:.4f} | Accuracy: {metrics['frame_accuracy']:.4f}")
    print(f"Note  F1: {metrics['note_f1']:.4f} | Precision: {metrics['note_precision']:.4f} "
      f"| Recall: {metrics['note_recall']:.4f}")


    # Save best model
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
