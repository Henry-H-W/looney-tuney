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
    Rows: Each row represents a unique pitch (same note does not change within row).
    Columns: 16th note subdivision in a bar -> SUBDIVISIONS
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
    """
    Converts parsed MIDI data into a matrix format:
    - Rows represent **a single pitch** (one row contains only one pitch at a time).
    - Columns represent **subdivisions within the bar** (16th notes).
    """
    bar_matrices = [np.zeros((NUM_KEYS, SUBDIVISIONS, 2)) for _ in range(num_bars)]
    
    # Keep track of row usage to avoid pitch changes in the same row
    row_pitch_mapping = {}

    for pitch, duration, velocity in notes_data:
        key_index = pitch - MIDI_LOW  # Convert pitch to index (A0 = index 0, C8 = index 87)
        
        # Assign a row if this pitch hasn't been used yet
        if pitch not in row_pitch_mapping:
            row_pitch_mapping[pitch] = key_index

        assigned_row = row_pitch_mapping[pitch]

        insert_note(bar_matrices, assigned_row, duration, velocity)

    return np.array(bar_matrices)

# Insert note into the correct matrix representation
def insert_note(bar_matrices, row_index, duration, velocity):
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
        Second Channel ([:,:,1]):
            Stores velocity.
        """
        if step == 0:
            bar_matrices[bar_idx][row_index, time_idx, 0] = 1  # Note onset
        else:
            bar_matrices[bar_idx][row_index, time_idx, 0] = 2  # Sustained note

        bar_matrices[bar_idx][row_index, time_idx, 1] = velocity  # Store velocity
        time_idx += 1

# Save the matrix representation to a file
def save_matrix_to_file(matrix_representation, filename="matrix_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:  # Specify UTF-8 encoding
        for bar_idx, bar in enumerate(matrix_representation):
            f.write(f"\nBar {bar_idx}:\n")  # Music emoji should now work
            for row in range(NUM_KEYS):
                row_data = " ".join([f"{int(bar[row, col, 0])}/{int(bar[row, col, 1])}" for col in range(SUBDIVISIONS)])
                f.write(row_data + "\n")
    print(f"Matrix saved to {filename}")

if __name__ == "__main__":
    midi_filename = "piano_right.mid"
    
    # Process MIDI file
    midi_data = parse_midi(midi_filename)
    matrix_representation = create_matrix_representation(midi_data)
    
    # Save to file
    save_matrix_to_file(matrix_representation)

    print("Matrix Representation Shape:", matrix_representation.shape)

    # Play MIDI file
    try:
        play_music(midi_filename)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
