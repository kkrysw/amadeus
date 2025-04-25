import os
from glob import glob
from tqdm import tqdm
import numpy as np
from pydub import AudioSegment
from mido import MidiFile
import sys
from joblib import Parallel, delayed
import multiprocessing

# === Auto-detect Google Drive path ===
def resolve_maps_path():
    gdrive_path = "/content/drive/MyDrive/Colab Notebooks/MAPS"
    local_link = "/content/MAPS"
    if not os.path.exists(local_link):
        os.symlink(gdrive_path, local_link)
    return local_link

MAPS_PATH = resolve_maps_path()
MIDI_PATH = os.path.join(MAPS_PATH, '*', 'MUS', '*.mid')
OUTPUT_TSV_DIR = os.path.join(MAPS_PATH, 'tsvs')

# === MIDI to TSV ===
def parse_midi(path):
    midi = MidiFile(path)
    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time
        if message.type == 'control_change' and message.control == 64:
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            events.append(dict(index=len(events), time=time, type=event_type, note=None, velocity=0))
        if 'note' in message.type:
            velocity = message.velocity if message.type == 'note_on' else 0
            events.append(dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain))

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue
        try:
            offset = next(n for n in events[i+1:] if n['note'] == onset['note'])
            if offset['sustain']:
                offset = next(n for n in events[offset['index']+1:] if n['type'] == 'sustain_off')
            notes.append((onset['time'], offset['time'], onset['note'], onset['velocity']))
        except StopIteration:
            continue

    return np.array(notes)

def process_midi(in_file, out_file):
    midi_data = parse_midi(in_file)
    np.savetxt(out_file, midi_data, fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')

def collect_files():
    midis = glob(MIDI_PATH)
    if not os.path.exists(OUTPUT_TSV_DIR):
        os.makedirs(OUTPUT_TSV_DIR)
    return [(m, os.path.join(OUTPUT_TSV_DIR, os.path.basename(m).replace('.mid', '.tsv'))) for m in midis]

midi_jobs = collect_files()
Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process_midi)(in_f, out_f) for in_f, out_f in midi_jobs)

print(f"MIDI -> TSV complete: {len(midi_jobs)} files written to {OUTPUT_TSV_DIR}")

# === WAV to FLAC ===
wavs = glob(os.path.join(MAPS_PATH, '**', '*.wav'), recursive=True)
for wav in tqdm(wavs, desc="Converting WAV to FLAC"):
    sound = AudioSegment.from_wav(wav)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(wav.replace('.wav', '.flac'), format='flac')

print(f"WAV -> FLAC conversion complete: {len(wavs)} files processed")

# === Dummy TSV for unmatched WAV ===
for wav in tqdm(wavs, desc="Generating dummy TSVs"):
    tsv_path = wav.replace('.wav', '.tsv')
    if not os.path.exists(tsv_path):
        notes = [(60, 60.5, 60, 64)] * 5
        np.savetxt(tsv_path, notes, fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')

print("Dummy TSVs generated where missing")
