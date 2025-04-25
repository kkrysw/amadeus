import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class PianoMAPSDataset(Dataset):
    def __init__(self, npz_dir, split='train', split_ratio=(0.8, 0.1, 0.1), seed=42):
        self.npz_dir = npz_dir
        self.file_paths = sorted(glob(os.path.join(npz_dir, '*.npz')))
        
        # Shuffle files consistently
        np.random.seed(seed)
        np.random.shuffle(self.file_paths)

        n_total = len(self.file_paths)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        if split == 'train':
            self.file_paths = self.file_paths[:n_train]
        elif split == 'val':
            self.file_paths = self.file_paths[n_train:n_train+n_val]
        elif split == 'test':
            self.file_paths = self.file_paths[n_train+n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz = np.load(self.file_paths[idx])
        mel = torch.tensor(npz['mel'], dtype=torch.float32).transpose(0, 1)  # [n_mels, time]
        label = torch.tensor(npz['label'], dtype=torch.float32).transpose(0, 1)  # [88, time]
        
        return mel, label
