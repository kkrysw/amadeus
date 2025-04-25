import os
from abc import abstractmethod
from glob import glob
import numpy as np
import torch
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi

class PianoRollAudioDataset(Dataset):
    _dataset_cache = {}

    def __init__(self, path, groups=None, sequence_length=None, seed=42, refresh=False, device='cpu'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.refresh = refresh

        cache_key = (self.__class__.__name__, tuple(sorted(self.groups)), path)
        if cache_key in PianoRollAudioDataset._dataset_cache and not self.refresh:
            print(f"Using cached data for {cache_key}")
            self.data = PianoRollAudioDataset._dataset_cache[cache_key]
        else:
            print(f"Loading data for {cache_key}")
            self.data = []
            for group in self.groups:
                for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                    self.data.append(self.load(*input_files))
            PianoRollAudioDataset._dataset_cache[cache_key] = self.data

    def __getitem__(self, index):
        result = {}
        max_tries = len(self.data)
        for _ in range(max_tries):
            data = self.data[index]
            if self.sequence_length is None or len(data['audio']) >= self.sequence_length:
                break
            index = (index + 1) % len(self.data)
        else:
            return None

        if self.sequence_length is not None:
            audio_length = len(data['audio'])
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps
            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = data['audio'][begin:end].to(self.device)
            result['label'] = data['label'][step_begin:step_end, :].to(self.device)
            result['velocity'] = data['velocity'][step_begin:step_end, :].to(self.device)
            result['start_idx'] = begin
        else:
            result['audio'] = data['audio'].to(self.device)
            result['label'] = data['label'].to(self.device)
            result['velocity'] = data['velocity'].to(self.device).float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        raise NotImplementedError

    def load(self, audio_path, tsv_path):
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path) and not self.refresh:
            return torch.load(saved_data_path)

        audio, sr = soundfile.read(audio_path, dtype='int16')
        assert sr == SAMPLE_RATE
        audio = torch.ShortTensor(audio)
        audio_length = len(audio)
        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        try:
            midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
            if midi.ndim == 1:
                midi = np.expand_dims(midi, axis=0)
        except Exception as e:
            print(f"[WARN] Could not read {tsv_path}: {e}")
            midi = np.zeros((0, 4))

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            label[left:onset_right, f] = 3
            label[onset_right:frame_right, f] = 2
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        return data


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='./MAPS', groups=None, sequence_length=None, overlap=True,
                 seed=42, refresh=False, device='cpu', supersmall=False):
        self.overlap = overlap
        self.supersmall = supersmall
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, refresh, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.flac') and f"_{group}" in file:
                    flac_path = os.path.join(root, file)
                    tsv_name = file.replace('.flac', '.tsv')
                    tsv_path = os.path.join(self.path, 'tsvs', tsv_name)
                    if not os.path.exists(tsv_path):
                        print(f"[WARN] Missing TSV for {flac_path}, skipping.")
                        continue
                    flacs.append((flac_path, tsv_path))

        assert all(os.path.isfile(f[0]) for f in flacs), "Some audio files missing."
        assert all(os.path.isfile(f[1]) for f in flacs), "Some TSVs missing."
        return sorted(flacs)
