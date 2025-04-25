import os
import torchaudio
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm

SAMPLE_RATE = 16000
N_MELS = 229
HOP_LENGTH = 512
MIN_MIDI = 21
MAX_MIDI = 108
N_NOTES = MAX_MIDI - MIN_MIDI + 1

def load_tsv(tsv_path):
    try:
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        if midi.ndim == 1:
            midi = np.expand_dims(midi, axis=0)
        return midi
    except Exception as e:
        print(f"⚠ Failed to load {tsv_path}: {e}")
        return np.zeros((0, 4))

def create_label_matrix(midi_data, n_frames):
    label = np.zeros((n_frames, N_NOTES), dtype=np.uint8)
    for onset, offset, note, vel in midi_data:
        if note < MIN_MIDI or note > MAX_MIDI:
            continue
        idx = int(note) - MIN_MIDI
        start = int(onset * SAMPLE_RATE // HOP_LENGTH)
        end = int(offset * SAMPLE_RATE // HOP_LENGTH)
        label[start:end, idx] = 1
    return label

def process_pair(audio_path, tsv_path, out_dir):
    audio, sr = sf.read(audio_path)
    if sr != SAMPLE_RATE:
        print(f"Resampling {audio_path} from {sr}Hz to {SAMPLE_RATE}Hz")
        audio = torchaudio.functional.resample(torch.tensor(audio), sr, SAMPLE_RATE).numpy()

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )(torch.tensor(audio).float()).numpy()

    mel_spec = mel_spec.T  # [time, n_mels]
    midi_data = load_tsv(tsv_path)
    label = create_label_matrix(midi_data, mel_spec.shape[0])

    save_path = os.path.join(out_dir, os.path.basename(audio_path).replace('.flac', '.npz'))
    np.savez(save_path, mel=mel_spec, label=label)

def main():
    audio_paths = glob('/content/MAPS/**/*.flac', recursive=True)
    out_dir = '/content/MAPS/npz'
    os.makedirs(out_dir, exist_ok=True)

    for audio_path in tqdm(audio_paths, desc="Processing MAPS audio"):
        tsv_path = audio_path.replace('.flac', '.tsv')
        if not os.path.exists(tsv_path):
            print(f"Missing TSV for {audio_path}, skipping.")
            continue
        try:
            process_pair(audio_path, tsv_path, out_dir)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    main()
