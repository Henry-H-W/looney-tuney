import os
import numpy as np
import pickle
from music21 import converter, note, chord, stream
from keras.models import load_model

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_FOLDER, "best_model.keras")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")  # Store generated MIDI files
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "generated.mid")

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Ensure the file exists.")

# Load trained notes
NOTES_PATH = os.path.join(BASE_DIR, "data", "notes")
if not os.path.exists(NOTES_PATH):
    raise FileNotFoundError(f"❌ Notes file not found at {NOTES_PATH}")

with open(NOTES_PATH, 'rb') as filepath:
    notes = pickle.load(filepath)

# Mapping for note conversion
pitchnames = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitchnames)}
int_to_note = {number: note for number, note in enumerate(pitchnames)}
n_vocab = len(pitchnames)

def get_notes(midi_file):
    """Extract notes from the given MIDI file"""
    notes = []
    midi = converter.parse(midi_file)
    notes_to_parse = midi.flatten()

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.pitches))

    return notes

def generate_music_from_midi(midi_file, output_file):
    """Generate music from a MIDI file"""
    if not os.path.exists(midi_file):
        raise FileNotFoundError(f"❌ MIDI file not found: {midi_file}")

    print(f"📥 Processing MIDI file: {midi_file}")

    # Extract notes from the given MIDI file
    notes = get_notes(midi_file)

    # Load trained model
    model = load_model(MODEL_PATH)
    print(f"✅ Model successfully loaded from: {MODEL_PATH}")

    # Convert input to sequences
    sequence_length = 100
    input_sequences = []
    
    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        input_sequences.append([note_to_int[note] for note in sequence_in])

    if not input_sequences:
        raise ValueError("❌ No valid sequences found in the MIDI file.")

    # Pick a random sequence to start generating from
    start = np.random.randint(0, len(input_sequences) - 1)
    pattern = input_sequences[start]
    prediction_output = []

    # Generate notes
    for _ in range(100):  
        input_sequence = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(input_sequence, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]

    # Convert predicted notes into a MIDI file
    output_midi = stream.Stream()
    for pattern in prediction_output:
        output_midi.append(note.Note(pattern, quarterLength=0.5))

    output_midi.write('midi', fp=output_file)
    print(f"✅ MIDI generated and saved at: {output_file}")
