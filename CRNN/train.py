import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from model.model import CRNN
from trueDataset import PianoMAPSDataset
from torch.utils.tensorboard import SummaryWriter

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
    csv.writer(f).writerow(['Epoch', 'Train Loss', 'Val Loss'])

best_val_loss = float('inf')

# --- Training Loop ---
for epoch in range(1, args.num_epochs + 1):
    model.train()
    train_loss = 0.0

    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        mel = mel.unsqueeze(1)  # [B, 1, 229, 512]

        optimizer.zero_grad()
        frame_out, onset_out = model(mel)

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
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            mel = mel.unsqueeze(1)  # [B, 1, 229, 512]

            frame_out, onset_out = model(mel)
            frame_loss = criterion(frame_out, label.clamp(0, 1))
            onset_loss = criterion(onset_out, (label > 0).float())
            val_loss += (frame_loss + onset_loss).item()

    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Save Best Model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
        print(f"New best model saved at epoch {epoch}")

    # --- Save Checkpoint ---
    if epoch % args.save_every == 0:
        ckpt_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

    # --- Log to TensorBoard + CSV ---
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Val', avg_val_loss, epoch)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss])

print("Training complete.")
writer.close()
