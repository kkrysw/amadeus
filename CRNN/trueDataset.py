import os
import torch
from torch.utils.data import Dataset

class PianoMAPSDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.input_dir = os.path.join(root_dir, 'mels')
        self.label_dir = os.path.join(root_dir, 'labels')

        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith("_mel.pt")])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith("_label.pt")])

        input_basenames = {f.replace("_mel.pt", ""): f for f in input_files}
        label_basenames = {f.replace("_label.pt", ""): f for f in label_files}

        common_basenames = sorted(set(input_basenames) & set(label_basenames))
        print(f"[DEBUG] Total input files: {len(common_basenames)}")

        self.data = []
        for base in common_basenames:
            input_path = os.path.join(self.input_dir, input_basenames[base])
            label_path = os.path.join(self.label_dir, label_basenames[base])
            self.data.append((input_path, label_path))

        print(f"[INFO] Collected {len(self.data)} input-label pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path, label_path = self.data[idx]
        input_tensor = torch.load(input_path).float().T.unsqueeze(0)  # [1, 229, T]
        label_tensor = torch.load(label_path).float()  # [T, 88]

        onset_tensor = (label_tensor[1:] > 0) & (label_tensor[:-1] == 0)
        onset_tensor = torch.cat([label_tensor[:1] > 0, onset_tensor], dim=0).float()  # [T, 88]

        return input_tensor, label_tensor, onset_tensor
