import os
import music21
from music21 import converter, note, chord
import pickle
import numpy as np
import keras
from keras.models import load_model

# Get the absolute path of the current script (generate.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the correct model path
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.keras")

# Ensure the model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Ensure the file exists.")

# Load the model
model = load_model(MODEL_PATH)
print(f"✅ Model successfully loaded from: {MODEL_PATH}")

# Constants
SEQUENCE_LENGTH = 100  
OUTPUT_FILE = 'server/output.mid'  # Save output in server directory

def get_notes(midi_file):
    """ Extract notes and chords from the given MIDI file """
    notes = []
    midi = converter.parse(midi_file)  # Read the uploaded MIDI file
    notes_to_parse = midi.flatten()
    
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.pitches))
    
    return notes

def generate_music_from_midi(midi_file, output_file):
    """ Generate music from a MIDI file and save output as MIDI """
    
    # Extract notes from the given MIDI file
    notes = get_notes(midi_file)

    # Create a mapping (note → integer)
    unique_notes = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(unique_notes)}
    int_to_note = {num: note for num, note in enumerate(unique_notes)}

    # Create input sequences for the model
    input_sequences = []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        input_sequences.append([note_to_int[char] for char in sequence_in])

    # Generate music sequence
    start = np.random.randint(0, len(input_sequences) - 1)
    pattern = input_sequences[start]
    prediction_output = []

    number_of_notes_to_generate = 100
    for i in range(number_of_notes_to_generate):
        input_sequence = np.reshape(pattern, (1, len(pattern), 1)) / float(len(unique_notes))
        prediction = model.predict(input_sequence, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    # Convert predicted notes into a MIDI file
    output_midi = music21.stream.Stream()
    shift = 0
    for pattern in prediction_output:
        if '.' in pattern:  # If it's a chord
            chord_notes = [music21.note.Note() for _ in pattern.split('.')]
            for i, note_in_chord in enumerate(pattern.split('.')):
                chord_notes[i].pitch = music21.pitch.Pitch(note_in_chord)
            output_midi.append(music21.chord.Chord(chord_notes, quarterLength=0.5))
        else:  # If it's a single note
            note_obj = music21.note.Note(pattern, quarterLength=0.5)
            output_midi.append(note_obj)

    # Save the generated MIDI file
    output_midi.write('midi', fp=output_file)
