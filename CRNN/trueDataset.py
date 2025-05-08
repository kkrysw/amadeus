import os
import torch
from torch.utils.data import Dataset

class PianoMAPSDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.input_dir = os.path.join(data_dir, "train_inputs")
        self.label_dir = os.path.join(data_dir, "train_labels")

        input_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith("_mel.pt")])
        label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith("_label.pt")])

        input_basenames = {f.replace("_mel.pt", ""): f for f in input_files}
        label_basenames = {f.replace("_label.pt", ""): f for f in label_files}
        common_basenames = sorted(set(input_basenames) & set(label_basenames))

        self.data = []
        skipped = 0
        for base in common_basenames:
            input_path = os.path.join(self.input_dir, input_basenames[base])
            label_path = os.path.join(self.label_dir, label_basenames[base])
            try:
                input_tensor = torch.load(input_path)
                label_tensor = torch.load(label_path)
                if input_tensor.shape[1] != label_tensor.shape[0]:
                    print(f"[SKIPPED] {base} - frame mismatch {input_tensor.shape[1]} vs {label_tensor.shape[0]}")
                    skipped += 1
                    continue
                self.data.append((input_path, label_path))
            except Exception as e:
                print(f"[SKIPPED] {base} - {e}")
                skipped += 1

        print(f"[INFO] Loaded {len(self.data)} valid samples. Skipped {skipped}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path, label_path = self.data[idx]
        input_tensor = torch.load(input_path).float()             # [1, 229, T]
        label_tensor = torch.load(label_path).float()             # [T, 88]

        # Derive onset: [T, 88]
        onset_tensor = (label_tensor[1:] > 0) & (label_tensor[:-1] == 0)
        onset_tensor = torch.cat([label_tensor[:1] > 0, onset_tensor], dim=0).float()

        return input_tensor, label_tensor, onset_tensor
