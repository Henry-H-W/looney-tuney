"""
    this is an OPTIONAL helper script in case your midi contains 
    multiple tracks that you'd like to combine into one!
"""

import os
import mido
from mido import MidiFile, MidiTrack, Message

# folder containing MIDI files
input_folder = r"dataset"
output_folder = "output_midis"
os.makedirs(output_folder, exist_ok=True)

# set the desired instrument (General MIDI program numbers: 0 = Acoustic Grand Piano)
program_number = 0  # change this to your desired instrument

def convert_midi_to_single_track(input_path, output_path, program):
    """Converts a MIDI file to a single-track, single-instrument version with proper timing."""
    midi = MidiFile(input_path)
    new_midi = MidiFile()
    new_track = MidiTrack()
    new_midi.tracks.append(new_track)

    # set the tempo based on the original MIDI file
    ticks_per_beat = midi.ticks_per_beat

    # store all events with absolute times
    events = []

    for track in midi.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type in ['note_on', 'note_off', 'control_change']:
                events.append((abs_time, msg))

    # sort events by absolute time to preserve timing order
    events.sort(key=lambda x: x[0])

    # write messages to the new track, adjusting delta times
    last_time = 0
    new_track.append(Message('program_change', program=program, time=0))

    for abs_time, msg in events:
        msg.time = abs_time - last_time  # convert absolute time back to delta time
        new_track.append(msg)
        last_time = abs_time

    new_midi.ticks_per_beat = ticks_per_beat
    new_midi.save(output_path)

# process all MIDI files in the input folder
for file in os.listdir(input_folder):
    if file.endswith(".mid") or file.endswith(".midi"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        convert_midi_to_single_track(input_path, output_path, program_number)
        print(f"Converted: {file} -> {output_path}")

print("All MIDI files converted successfully with correct timing!")
