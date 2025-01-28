import pygame
from music21 import converter, note, chord
import numpy as np

# Initialize pygame mixer
freq = 44100  # Audio CD quality
bitsize = -16  # Unsigned 16 bit
channels = 2  # Stereo
buffer = 1024  # Number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

def play_music(midi_filename):
    """Stream MIDI file in a blocking manner."""
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

# MIDI Preprocessing
"""
Goal: 
1D Array: Represents the sequence of bars (length 16)

2D Matrices (Bars of Music):
    Rows (88): Specific piano key (from A0 to C8) -> NUM_KEYS
    Columns (16): 16th note subdivision in a bar -> SUBDIVISIONS
    Each Cell: Stores information about a note event
        Onset (1 if note starts, 0 otherwise)
        Sustain (indicates continued note hold)
        Velocity (volume of the note)
"""

NUM_KEYS = 88  # Number of piano keys
SUBDIVISIONS = 16  # 16th note divisions per bar
MIDI_LOW = 21  # MIDI number for A0 (lowest piano key)
DEFAULT_BARS = 16  # Default number of bars to store

# Parse MIDI data
def parse_midi(midi_file):
    midi_data = converter.parse(midi_file)
    notes_data = []

    for part in midi_data.parts:
        for element in part.flat.notes:
            if isinstance(element, note.Note):
                pitch = element.pitch.midi  # Convert pitch to MIDI number
                duration = element.quarterLength  # Note duration in beats
                velocity = element.volume.velocity if element.volume.velocity else 64  # Default velocity if None
                notes_data.append((pitch, duration, velocity))
            elif isinstance(element, chord.Chord):
                pitches = [p.midi for p in element.pitches]
                duration = element.quarterLength
                velocity = element.volume.velocity if element.volume.velocity else 64
                for p in pitches:
                    notes_data.append((p, duration, velocity))
    
    return notes_data

# Convert MIDI notes into 2D matrices
def create_matrix_representation(notes_data, num_bars=DEFAULT_BARS):
    bar_matrices = [np.zeros((NUM_KEYS, SUBDIVISIONS, 2)) for _ in range(num_bars)]

    for pitch, duration, velocity in notes_data:
        key_index = pitch - MIDI_LOW  # Convert pitch to index (A0 = index 0, C8 = index 87)
        insert_note(bar_matrices, key_index, duration, velocity)

    return np.array(bar_matrices)

# Insert note into the correct matrix representation
def insert_note(bar_matrices, key_index, duration, velocity):
    bar_idx = 0  # Start with the first bar
    time_idx = 0  # Start at the beginning of the bar
    note_length = int(duration * SUBDIVISIONS / 4)  # Convert beats to 16th notes

    for step in range(note_length):
        if time_idx >= SUBDIVISIONS:
            bar_idx += 1  # Move to next bar
            time_idx = 0
            if bar_idx >= len(bar_matrices):
                break  # Stop if out of bars

        """
        For each cell stores information about a note event
        First Channel ([:,:,0]):
            1: Note onset.
            2: Sustained note.
            0: No note.
        Second Channel ([:,:,1]
        """
        if step == 0:
            bar_matrices[bar_idx][key_index, time_idx, 0] = 1  # Note onset
        else:
            bar_matrices[bar_idx][key_index, time_idx, 0] = 2  # Sustained note

        bar_matrices[bar_idx][key_index, time_idx, 1] = velocity  # Store velocity
        time_idx += 1

if __name__ == "__main__":
    midi_filename = "data/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi"
    
    # Play MIDI file
    try:
        play_music(midi_filename)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

    # Process MIDI file
    midi_data = parse_midi(midi_filename)
    matrix_representation = create_matrix_representation(midi_data)
    
    print("Matrix Representation Shape:", matrix_representation.shape)