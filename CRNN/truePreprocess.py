import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment
from mido import MidiFile
from joblib import Parallel, delayed
import multiprocessing
import shutil

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# === Parameters ===
BASE_DIR = r"C:\Users\kevin\Downloads\MAPS"
TEMP_DIR = os.path.join(BASE_DIR, "temp_extract")
OUTPUT_DIR = os.path.join(BASE_DIR, "newMAP")
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_TSV_DIR = os.path.join(OUTPUT_DIR, "tsvs")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_TSV_DIR, exist_ok=True)

# === MIDI to TSV Functions ===
def parse_midi(path):
    from miditoolkit import MidiFile

    midi = MidiFile(path)
    notes = []

    for note in midi.instruments[0].notes:
        onset = note.start / midi.ticks_per_beat
        offset = note.end / midi.ticks_per_beat
        pitch = note.pitch
        velocity = note.velocity
        notes.append((onset, offset, pitch, velocity))

    if len(notes) == 0:
        print(f"[DEBUG] No notes in {os.path.basename(path)}")

    return np.array(notes)


def save_tsv(input_file, output_file):
    try:
        data = parse_midi(input_file)
        if data.size > 0:
            np.savetxt(output_file, data, fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')
    except Exception as e:
        print(f"[MIDI ERROR] Failed to parse {input_file}: {e}")

# === Find all MIDI and WAV files recursively ===
midi_files = glob(os.path.join(BASE_DIR, '**', '*.mid'), recursive=True)
wav_files = glob(os.path.join(BASE_DIR, '**', '*.wav'), recursive=True)

print(f"[INFO] Found {len(midi_files)} MIDI files.")
print(f"[INFO] Found {len(wav_files)} WAV files.")

# === Process MIDI to TSV ===
def process_midi_file(midi_path):
    output_path = os.path.join(OUTPUT_TSV_DIR, os.path.basename(midi_path).replace('.mid', '.tsv'))
    save_tsv(midi_path, output_path)

Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(process_midi_file)(m) for m in tqdm(midi_files, desc="Processing MIDI files")
)

# === Convert WAV to FLAC ===
for wav_path in tqdm(wav_files, desc="Converting WAV to FLAC"):
    try:
        sound = AudioSegment.from_wav(wav_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        flac_name = os.path.basename(wav_path).replace('.wav', '.flac')
        output_flac_path = os.path.join(OUTPUT_AUDIO_DIR, flac_name)
        sound.export(output_flac_path, format='flac')
    except Exception as e:
        print(f"[WAV ERROR] Failed to convert {wav_path}: {e}")

# === Match TSVs to FLACs ===
for tsv_path in glob(os.path.join(OUTPUT_TSV_DIR, "*.tsv")):
    base_name = os.path.basename(tsv_path).replace('.tsv', '')
    flac_match = os.path.join(OUTPUT_AUDIO_DIR, base_name + ".flac")
    if os.path.exists(flac_match):
        shutil.move(tsv_path, os.path.join(OUTPUT_AUDIO_DIR, os.path.basename(tsv_path)))
    else:
        print(f"[WARN] No matching FLAC found for: {base_name}. TSV kept in tsvs/ folder.")

# === Zip final output ===
shutil.make_archive(os.path.join(BASE_DIR, "newMAP"), 'zip', OUTPUT_DIR)
print("[DONE] newMAP.zip created successfully.")
