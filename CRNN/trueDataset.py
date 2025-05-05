import os
import torch
from torch.utils.data import Dataset

class PianoMAPSDataset(Dataset):
    def __init__(self, tensor_dir, split='train'):
        assert split in ['train', 'val'], f"Invalid split: {split}"
        self.input_dir = os.path.join(tensor_dir, f"{split}_inputs")
        self.label_dir = os.path.join(tensor_dir, f"{split}_labels")

        # Match *_mel.pt and *_label.pt files
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('_mel.pt')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('_label.pt')])

        # Map base name to file
        self.basename_to_input = {f.replace('_mel.pt', ''): f for f in self.input_files}
        self.basename_to_label = {f.replace('_label.pt', ''): f for f in self.label_files}

        self.common_keys = sorted(set(self.basename_to_input.keys()) & set(self.basename_to_label.keys()))
        if len(self.common_keys) == 0:
            raise RuntimeError("No matching input/label tensor pairs found.")

        # Sanity check on one sample
        sample_input = torch.load(os.path.join(self.input_dir, self.basename_to_input[self.common_keys[0]]))
        sample_label = torch.load(os.path.join(self.label_dir, self.basename_to_label[self.common_keys[0]]))
        assert sample_input.ndim == 2 and sample_label.ndim == 2, "Expected 2D tensors [freq_bins, frames]"
        assert sample_input.shape[1] == sample_label.shape[0], \
            f"Mismatch: input frames {sample_input.shape[1]} vs label frames {sample_label.shape[0]}"

    def __len__(self):
        return len(self.common_keys)

    def __getitem__(self, idx):
        base = self.common_keys[idx]
        input_tensor = torch.load(os.path.join(self.input_dir, self.basename_to_input[base])).float()
        label_tensor = torch.load(os.path.join(self.label_dir, self.basename_to_label[base])).float()
        return input_tensor.unsqueeze(0), label_tensor
