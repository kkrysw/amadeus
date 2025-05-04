import os
import torch
from torch.utils.data import Dataset
import numpy as np

class PianoMAPSDataset(Dataset):
    def __init__(self, tensor_dir, split='train'):
        assert split in ['train', 'val'], f"Invalid split: {split}"
        self.tensor_dir = tensor_dir
        self.split = split

        self.input_dir = os.path.join(tensor_dir, f"{split}_inputs")
        self.label_dir = os.path.join(tensor_dir, f"{split}_labels")

        self.files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.pt')])
        if len(self.files) == 0:
            raise RuntimeError(f"No tensor files found in {self.input_dir}")

        # Sanity check on dimensions
        sample_input = torch.load(os.path.join(self.input_dir, self.files[0]))
        sample_label = torch.load(os.path.join(self.label_dir, self.files[0]))
        assert sample_input.ndim == 2 and sample_label.ndim == 2, "Expected 2D tensors [freq_bins, frames]"
        assert sample_input.shape[1] == sample_label.shape[0], \
            f"Input frames {sample_input.shape[1]} and label frames {sample_label.shape[0]} mismatch."

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        input_tensor = torch.load(os.path.join(self.input_dir, fname)).float()
        label_tensor = torch.load(os.path.join(self.label_dir, fname)).float()
        # Reshape: [freq_bins, frames] → [1, freq_bins, frames] to treat as a channel
        return input_tensor.unsqueeze(0), label_tensor
