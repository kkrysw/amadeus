import os
import torch
from torch.utils.data import Dataset

class PianoMAPSDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.input_dir = f"/content/train_inputs"
        self.label_dir = f"/content/train_labels"
        self.input_paths = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.pt')])
        self.label_paths = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.pt')])

        self.data = []
        skipped = 0

        for f in self.input_paths:
            input_path = os.path.join(self.input_dir, f)
            label_path = os.path.join(self.label_dir, f)
            if not os.path.exists(label_path):
                skipped += 1
                continue

            try:
                sample_input = torch.load(input_path)
                sample_label = torch.load(label_path)

                # Check shape compatibility
                assert sample_input.shape[1] == sample_label.shape[0], \
                    f"Mismatch: input frames {sample_input.shape[1]} vs label frames {sample_label.shape[0]}"
                self.data.append((input_path, label_path))
            except Exception as e:
                print(f"[SKIPPED] {f} - {e}")
                skipped += 1

        print(f"[INFO] Loaded {len(self.data)} valid samples. Skipped {skipped}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        base = self.data[idx]
        input_tensor = torch.load(os.path.join(self.input_dir, self.basename_to_input[base])).float()
        label_tensor = torch.load(os.path.join(self.label_dir, self.basename_to_label[base])).float()
        return input_tensor.unsqueeze(0), label_tensor
