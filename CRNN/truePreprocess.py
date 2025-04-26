import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment
from mido import MidiFile
from joblib import Parallel, delayed
import multiprocessing
import shutil

# === Parameters ===
BASE_DIR = r"C:\Users\AlexWu\Documents\DeepLearning\Porject Shit\MAPS"
TEMP_DIR = os.path.join(BASE_DIR, "temp_extract")
OUTPUT_DIR = os.path.join(BASE_DIR, "newMAP")
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
OUTPUT_TSV_DIR = os.path.join(OUTPUT_DIR, "tsvs")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_TSV_DIR, exist_ok=True)

# === MIDI to TSV Functions ===
def parse_midi(path):
    midi = MidiFile(path)
    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time
        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            sustain = message.value >= 64
            event = dict(index=len(events), time=time, type='sustain_on' if sustain else 'sustain_off', note=None, velocity=0)
            events.append(event)
        if 'note' in message.type:
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)
    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue
        try:
            offset = next(n for n in events[i+1:] if n['note'] == onset['note'] or n is events[-1])
            if offset['sustain'] and offset is not events[-1]:
                offset = next(n for n in events[offset['index']+1:] if n['type'] == 'sustain_off' or (n.get('note') == offset['note']) or n is events[-1])
            note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
            notes.append(note)
        except Exception as e:
            print(f"Skipping note due to error: {e}")
            continue
    return np.array(notes)

def save_tsv(input_file, output_file):
    try:
        data = parse_midi(input_file)
        if data.size > 0:
            np.savetxt(output_file, data, fmt='%.6f', delimiter='\t', header='onset\toffset\tnote\tvelocity')
    except Exception as e:
        print(f"Error parsing {input_file}: {e}")

# === Find all MIDI and WAV files recursively ===
midi_files = glob(os.path.join(BASE_DIR, '**', '*.mid'), recursive=True)
wav_files = glob(os.path.join(BASE_DIR, '**', '*.wav'), recursive=True)

print(f"Found {len(midi_files)} MIDI files.")
print(f"Found {len(wav_files)} WAV files.")

# === Process MIDI to TSV ===
def process_midi_file(midi_path):
    output_path = os.path.join(OUTPUT_TSV_DIR, os.path.basename(midi_path).replace('.mid', '.tsv'))
    save_tsv(midi_path, output_path)

Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(process_midi_file)(m) for m in midi_files
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
        print(f"Failed to convert {wav_path}: {e}")

# === Match TSVs to FLACs ===
for tsv_path in glob(os.path.join(OUTPUT_TSV_DIR, "*.tsv")):
    base_name = os.path.basename(tsv_path).replace('.tsv', '')
    flac_match = os.path.join(OUTPUT_AUDIO_DIR, base_name + ".flac")
    if os.path.exists(flac_match):
        shutil.move(tsv_path, os.path.join(OUTPUT_AUDIO_DIR, os.path.basename(tsv_path)))
    else:
        print(f"No matching FLAC found for {base_name}, keeping TSV in output folder.")

print("Preprocessing complete. All TSVs and FLACs are ready in the 'newMAP' folder.")

# === Zip final output ===
shutil.make_archive(os.path.join(BASE_DIR, "newMAP"), 'zip', OUTPUT_DIR)

print("newMAP.zip created successfully.")
