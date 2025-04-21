from scipy.io import wavfile
from pydub import AudioSegment
from glob import glob
from tqdm import tqdm
import numpy as np
import os

# libraries and functions for midi2tsv
import multiprocessing
import sys
import mido
import numpy as np
from joblib import Parallel, delayed
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi

def parse_midi(path):
    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
#             offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n['note'] == offset['note'] or n is events[-1])
        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


def save_midi(path, pitches, intervals, velocities):
    """
    Save extracted notes as a MIDI file
    Parameters
    ----------
    path: the path to save the MIDI file
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)

def process(input_file, output_file):
    midi_data = parse_midi(input_file)
    np.savetxt(output_file, midi_data, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')


def files(file_list, output_dir=False):
    for input_file in tqdm(file_list):
        if input_file.endswith('.mid'):
            if output_dir==False:
                output_file = input_file[:-4] + '.tsv'
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file[:-4]) + '.tsv')
        elif input_file.endswith('.midi'):
            if output_dir==False:
                output_file = input_file[:-5] + '.tsv'
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file[:-5]) + '.tsv')                
        else:
            print('ignoring non-MIDI file %s' % input_file, file=sys.stderr)
            continue

        yield (input_file, output_file)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

midis = glob('./MAPS/*/MUS/*.mid') # loading lists of midi files
output_dir = '../MAPS/tsvs' # prepare a dir for the tsv output
if os.path.exists(output_dir):
    pass
else:
    os.makedirs(output_dir)
Parallel(n_jobs=multiprocessing.cpu_count())(delayed(process)(in_file, out_file) for in_file, out_file in files(midis, output_dir=output_dir))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

for wavfile in tqdm(glob('.././*/*.wav')):
    sound = AudioSegment.from_wav(wavfile)
    sound = sound.set_frame_rate(16000) # downsample it to 16000
    sound = sound.set_channels(1) # Convert Stereo to Mono
    
    sound.export(wavfile[:-3] + 'flac', format='flac')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Creating dummy tsv for the VAT
for wavfile in tqdm(glob('.././*/*.wav')):
    tsv_path = wavfile.replace('.wav', '.tsv')
    
    notes = []
    note = (60,60,60,60)
    for i in range(5):
        notes.append(note)
    
    np.savetxt(tsv_path, notes, '%.6f', '\t', header='onset\toffset\tnote\tvelocity')
