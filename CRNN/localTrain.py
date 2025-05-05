import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from model.model import CRNN
from localTrueDataset import LocalPianoMAPSDataset

def compute_frame_metrics(preds, targets, threshold=0.1):
    preds_bin = (preds > threshold).cpu().numpy().astype(int)
    targets_bin = (targets > 0.5).cpu().numpy().astype(int)
    p, r, f1, _ = precision_recall_fscore_support(targets_bin.flatten(), preds_bin.flatten(), average='binary', zero_division=0)
    acc = accuracy_score(targets_bin.flatten(), preds_bin.flatten())
    return {"frame_precision": p, "frame_recall": r, "frame_f1": f1, "frame_accuracy": acc}

# Set local tensor path
tensor_dir = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS\preprocessed_tensors"
save_dir = "./weights_local"
os.makedirs(save_dir, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CRNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

train_loader = DataLoader(LocalPianoMAPSDataset(tensor_dir, 'train'), batch_size=8, shuffle=True)
val_loader = DataLoader(LocalPianoMAPSDataset(tensor_dir, 'val'), batch_size=8, shuffle=False)

csv_path = os.path.join(save_dir, 'loss_log.csv')
with open(csv_path, 'w', newline='') as f:
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss', 'F1', 'Precision', 'Recall', 'Accuracy'])

for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for mel, label in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        frame_out, onset_out = model(mel)
        loss = criterion(frame_out, label) + 5.0 * criterion(onset_out, label)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss, all_preds, all_targets = 0, [], []
    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            frame_out, onset_out = model(mel)
            loss = criterion(frame_out, label) + 5.0 * criterion(onset_out, label)
            val_loss += loss.item()
            all_preds.append(torch.sigmoid(frame_out).cpu())
            all_targets.append(label.cpu())

    avg_val_loss = val_loss / len(val_loader)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_frame_metrics(preds, targets)

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Frame F1: {metrics['frame_f1']:.4f} | Precision: {metrics['frame_precision']:.4f} | Recall: {metrics['frame_recall']:.4f}")

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([
            epoch, avg_train_loss, avg_val_loss,
            metrics['frame_f1'], metrics['frame_precision'],
            metrics['frame_recall'], metrics['frame_accuracy']
        ])

    scheduler.step()

print(" Local training finished.")
