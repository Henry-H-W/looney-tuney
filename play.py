import os
from pypianoroll import read, to_pretty_midi
from pydub import AudioSegment, playback
import soundfile as sf
import numpy as np
import pretty_midi

# Define sample rate for smooth playback
sample_rate = 44100

# Find the most recent MIDI file in the directory
midi_files = [f for f in os.listdir() if f.endswith(".mid")]
if not midi_files:
    raise FileNotFoundError("No MIDI files found in the directory.")

# Get the most recently created/modified MIDI file
latest_midi = max(midi_files, key=os.path.getctime)
original_file_name = os.path.splitext(latest_midi)[0]  # Remove .mid extension

print(f"Processing most recent MIDI file: {latest_midi}")

# Load MIDI file
multitrack = read(latest_midi)

# Convert to PrettyMIDI
pm = to_pretty_midi(multitrack)

# Create a new PrettyMIDI object
new_pm = pretty_midi.PrettyMIDI()

# MIDI Velocity Ranges and Corresponding Dynamic Levels:
# -----------------------------------------------
# Velocity  |  Dynamic Level
# -----------------------------------------------
#  0  -  20  |  ppp (very soft)
# 21  -  40  |  pp  (soft)
# 41  -  60  |  p   (moderately soft)
# 61  -  80  |  mp  (medium soft)
# 81  - 100  |  mf  (medium loud)
# 101 - 120  |  f   (loud)
# 121 - 127  |  ff-fff (very loud)
# -----------------------------------------------

for instrument in pm.instruments:
    new_instrument = pretty_midi.Instrument(program=instrument.program)
    
    # Sort notes by start time
    instrument.notes.sort(key=lambda note: note.start)
    
    # Process notes with arpeggiation only for long chords
    last_start_time = -1
    chord_notes = []

    for note in instrument.notes:
        new_velocity = 20  # Set all notes to velocity 20 (ppp)

        # Introduce timing variation to all notes (-10ms to +10ms)
        time_variation = np.random.uniform(-0.01, 0.01)
        new_start = max(0, note.start + time_variation)
        new_end = max(new_start + 0.05, note.end + time_variation)

        # Group notes that start at the same time (potential chords)
        if np.isclose(note.start, last_start_time, atol=0.001):
            chord_notes.append(note)
        else:
            # Apply arpeggiation if the chord is held for at least 1 second
            if chord_notes and all(n.end - n.start >= 1.0 for n in chord_notes):
                for i, chord_note in enumerate(chord_notes):
                    time_offset = i * np.random.uniform(0.02, 0.04)  # Slight delay (20ms to 50ms)
                    new_start = max(0, chord_note.start + time_offset)
                    new_end = max(new_start + 0.05, chord_note.end + time_offset)

                    new_note = pretty_midi.Note(
                        velocity=20,  # Set velocity to 20 (ppp)
                        pitch=chord_note.pitch,
                        start=new_start,
                        end=new_end
                    )
                    new_instrument.notes.append(new_note)
            else:
                # Apply timing variation to short chords and normal notes
                for chord_note in chord_notes:
                    time_variation = np.random.uniform(-0.01, 0.01)  # Small humanization shift
                    new_start = max(0, chord_note.start + time_variation)
                    new_end = max(new_start + 0.05, chord_note.end + time_variation)

                    new_note = pretty_midi.Note(
                        velocity=20,  # Set velocity to 20 (ppp)
                        pitch=chord_note.pitch,
                        start=new_start,
                        end=new_end
                    )
                    new_instrument.notes.append(new_note)

            # Reset for the new chord
            chord_notes = [note]
            last_start_time = note.start

    # Process the last chord (if it's long enough)
    if chord_notes and all(n.end - n.start >= 1.0 for n in chord_notes):
        for i, chord_note in enumerate(chord_notes):
            time_offset = i * np.random.uniform(0.02, 0.04)  # Slight delay
            new_start = max(0, chord_note.start + time_offset)
            new_end = max(new_start + 0.05, chord_note.end + time_offset)

            new_note = pretty_midi.Note(
                velocity=20,  # Set velocity to 20 (ppp)
                pitch=chord_note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)
    else:
        # Apply timing variation to the last set of notes
        for chord_note in chord_notes:
            time_variation = np.random.uniform(-0.01, 0.01)  # Small humanization shift
            new_start = max(0, chord_note.start + time_variation)
            new_end = max(new_start + 0.05, chord_note.end + time_variation)

            new_note = pretty_midi.Note(
                velocity=20,  # Set velocity to 20 (ppp)
                pitch=chord_note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)

    new_pm.instruments.append(new_instrument)

# Save the modified MIDI file explicitly
new_midi_path = f"edited_{original_file_name}.mid"
new_pm.write(new_midi_path)

# Load the new MIDI file to ensure changes are applied
pm_humanized = pretty_midi.PrettyMIDI(new_midi_path)

# Convert new MIDI to waveform
wave = pm_humanized.fluidsynth(sf2_path="SalC5Light2.sf2", fs=sample_rate)

# Save as WAV for pydub
sf.write("editted_midi.wav", wave, sample_rate)

# Load into pydub
segment = AudioSegment.from_wav("editted_midi.wav")

# Apply a low-pass filter (reduce brightness)
filtered_segment = segment.low_pass_filter(2500)

# Add subtle reverb
filtered_segment = filtered_segment.overlay(filtered_segment - 6, position=50)

# Apply compression (avoid over-amplification of quiet notes)
filtered_segment = filtered_segment.normalize()

# Play the final audio
playback.play(filtered_segment)