import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import CRNN
from trueDataset import PianoMAPSDataset

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train CRNN on MAPS Dataset")
parser.add_argument('--data_dir', type=str, required=True, help='Path to audio + tsv files')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--save_dir', type=str, default='./weights')
args = parser.parse_args()

# --- Config ---
DATA_PATH = args.data_dir
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
SAVE_DIR = args.save_dir
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Data ---
train_dataset = PianoMAPSDataset(DATA_PATH, split='train')
val_dataset = PianoMAPSDataset(DATA_PATH, split='val')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- Initialize Model ---
model = CRNN()
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# --- Training Loop ---
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()
        frame_out, onset_out = model(mel)

        frame_loss = criterion(frame_out, label)
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

            frame_out, onset_out = model(mel)

            frame_loss = criterion(frame_out, label)
            onset_loss = criterion(onset_out, (label > 0).float())
            loss = frame_loss + onset_loss

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Save best model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = os.path.join(SAVE_DIR, 'best_model.pt')
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model at epoch {epoch} to {save_path}")

print("Training finished.")
