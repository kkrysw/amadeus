import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf

class PianoMAPSDataset(Dataset):
    def __init__(self, data_dir, split='train', split_ratio=(0.8, 0.1, 0.1), seed=42, sr=16000, n_mels=229):
        self.data_dir = data_dir
        self.audio_paths = sorted([f for f in os.listdir(data_dir) if f.endswith('.flac')])

        # Consistent shuffling
        np.random.seed(seed)
        np.random.shuffle(self.audio_paths)

        # Train/val/test split
        n_total = len(self.audio_paths)
        n_train = int(split_ratio[0] * n_total)
        n_val = int(split_ratio[1] * n_total)

        if split == 'train':
            self.audio_paths = self.audio_paths[:n_train]
        elif split == 'val':
            self.audio_paths = self.audio_paths[n_train:n_train+n_val]
        elif split == 'test':
            self.audio_paths = self.audio_paths[n_train+n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_filename = self.audio_paths[idx]
        audio_path = os.path.join(self.data_dir, audio_filename)
        tsv_path = audio_path.replace('.flac', '.tsv')

        # Load audio
        y, _ = sf.read(audio_path)
        y = librosa.resample(y, orig_sr=self.sr, target_sr=self.sr)
        
        # Compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels)
        mel = librosa.power_to_db(mel, ref=np.max)

        # Load label
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Missing label file: {tsv_path}")

        labels = np.loadtxt(tsv_path, skiprows=1, delimiter='\t')

        # Build piano roll (binary frame labels)
        time_steps = mel.shape[1]
        frame_labels = np.zeros((88, time_steps), dtype=np.float32)

        duration = len(y) / self.sr
        frame_times = np.linspace(0, duration, num=time_steps)

        for onset, offset, note, velocity in labels:
            note_idx = int(note) - 21  # MIDI note 21 = A0
            if 0 <= note_idx < 88:
                active = (frame_times >= onset) & (frame_times <= offset)
                frame_labels[note_idx, active] = velocity / 127.0  # Normalize velocity

        mel = torch.tensor(mel, dtype=torch.float32)
        label = torch.tensor(frame_labels, dtype=torch.float32)

        return mel, label
