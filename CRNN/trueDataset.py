import os
import torch
from torch.utils.data import Dataset

class PianoMAPSDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.mel_batches = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.startswith("mel_batch")])
        self.label_batches = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.startswith("label_batch")])

        assert len(self.mel_batches) == len(self.label_batches), "Mismatch between mel and label batch files"
        print(f"[INFO] Found {len(self.mel_batches)} batched mel/label pairs.")

        self.batch_index_map = []
        self.loaded_batches = {}

        for batch_idx, mel_path in enumerate(self.mel_batches):
            mel_tensor = torch.load(mel_path, map_location="cpu")
            num_samples = mel_tensor.shape[0]
            for i in range(num_samples):
                self.batch_index_map.append((batch_idx, i))

    def __len__(self):
        return len(self.batch_index_map)

    def __getitem__(self, idx):
        batch_idx, sample_idx = self.batch_index_map[idx]

        if batch_idx not in self.loaded_batches:
            mel = torch.load(self.mel_batches[batch_idx], map_location="cpu")
            label = torch.load(self.label_batches[batch_idx], map_location="cpu")
            self.loaded_batches[batch_idx] = (mel, label)

        mel_batch, label_batch = self.loaded_batches[batch_idx]
        mel_tensor = mel_batch[sample_idx].unsqueeze(0)  # [1, T, 128]
        label_tensor = label_batch[sample_idx]           # [T, 88]

        onset_tensor = (label_tensor[1:] > 0) & (label_tensor[:-1] == 0)
        onset_tensor = torch.cat([label_tensor[:1] > 0, onset_tensor], dim=0).float()

        return mel_tensor, label_tensor, onset_tensor
