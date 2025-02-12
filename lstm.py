# lstm.py
import glob
import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

# TensorFlow / Keras imports
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def get_notes():
    """
    Parse all MIDI files in the 'data' folder and extract pitch and duration for each note/chord.
    Each note is stored as a tuple: (pitch, duration)
      - For a note: pitch is a string like "C4".
      - For a chord: pitch is a string of note integers joined by dots (e.g., "4.7.11").
      - The duration is stored as a string representing the quarter length (rounded to 2 decimals).
    """
    notes = []
    midi_files = glob.glob("data/*.mid") + glob.glob("data/*.midi")
    if not midi_files:
        raise Exception("No MIDI files found in the 'data' folder.")
    for file in midi_files:
        print(f"Parsing {file}...")
        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            dur = str(round(element.duration.quarterLength, 2))
            if isinstance(element, note.Note):
                pitch = str(element.pitch)
                notes.append((pitch, dur))
            elif isinstance(element, chord.Chord):
                pitch = '.'.join(str(n) for n in element.normalOrder)
                notes.append((pitch, dur))
    return notes

def prepare_sequences(notes, sequence_length):
    """
    Prepare training sequences from the notes data.
    Returns:
       - X_pitch: sequences of pitch indices (shape: [num_samples, sequence_length])
       - X_duration: sequences of duration indices (shape: [num_samples, sequence_length])
       - y_pitch: one-hot targets for pitch (shape: [num_samples, n_pitch])
       - y_duration: one-hot targets for duration (shape: [num_samples, n_duration])
       - Dictionaries for pitch and duration conversion along with other parameters.
    """
    # Separate pitches and durations
    pitches = [n[0] for n in notes]
    durations = [n[1] for n in notes]
    
    # Unique values and their counts
    unique_pitches = sorted(set(pitches))
    unique_durations = sorted(set(durations))
    
    n_pitch = len(unique_pitches)
    n_duration = len(unique_durations)
    
    # Create mapping dictionaries
    pitch_to_int = {p: i for i, p in enumerate(unique_pitches)}
    int_to_pitch = {i: p for i, p in enumerate(unique_pitches)}
    
    duration_to_int = {d: i for i, d in enumerate(unique_durations)}
    int_to_duration = {i: d for i, d in enumerate(unique_durations)}
    
    network_input_pitch = []
    network_input_duration = []
    network_output_pitch = []
    network_output_duration = []
    
    for i in range(0, len(notes) - sequence_length):
        seq_in_pitch = [pitch_to_int[notes[j][0]] for j in range(i, i+sequence_length)]
        seq_in_duration = [duration_to_int[notes[j][1]] for j in range(i, i+sequence_length)]
        seq_out_pitch = pitch_to_int[notes[i+sequence_length][0]]
        seq_out_duration = duration_to_int[notes[i+sequence_length][1]]
        network_input_pitch.append(seq_in_pitch)
        network_input_duration.append(seq_in_duration)
        network_output_pitch.append(seq_out_pitch)
        network_output_duration.append(seq_out_duration)
    
    n_patterns = len(network_input_pitch)
    
    # Convert to numpy arrays and one-hot encode the outputs
    X_pitch = np.array(network_input_pitch)
    X_duration = np.array(network_input_duration)
    y_pitch = to_categorical(network_output_pitch, num_classes=n_pitch)
    y_duration = to_categorical(network_output_duration, num_classes=n_duration)
    
    return (X_pitch, X_duration, y_pitch, y_duration, 
            pitch_to_int, int_to_pitch, duration_to_int, int_to_duration, 
            sequence_length, n_pitch, n_duration)

def create_network(sequence_length, n_pitch, n_duration):
    """
    Create the multi-channel LSTM network with two inputs (pitch and duration) and two outputs.
    """
    # Define inputs for pitch and duration (each as a sequence of integers)
    input_pitch = Input(shape=(sequence_length,), name="pitch_input")
    input_duration = Input(shape=(sequence_length,), name="duration_input")
    
    # Embedding layers for each input
    embed_pitch = Embedding(input_dim=n_pitch, output_dim=100, input_length=sequence_length)(input_pitch)
    embed_duration = Embedding(input_dim=n_duration, output_dim=20, input_length=sequence_length)(input_duration)
    
    # Concatenate the embeddings along the feature axis
    merged = Concatenate()([embed_pitch, embed_duration])
    
    # Shared LSTM layers
    lstm1 = LSTM(256, return_sequences=True)(merged)
    drop1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(256)(drop1)
    drop2 = Dropout(0.3)(lstm2)
    
    # Branch for pitch prediction
    dense_pitch = Dense(256, activation='relu')(drop2)
    pitch_output = Dense(n_pitch, activation='softmax', name="pitch_output")(dense_pitch)
    
    # Branch for duration prediction
    dense_duration = Dense(64, activation='relu')(drop2)
    duration_output = Dense(n_duration, activation='softmax', name="duration_output")(dense_duration)
    
    model = Model(inputs=[input_pitch, input_duration], outputs=[pitch_output, duration_output])
    model.compile(loss={'pitch_output': 'categorical_crossentropy', 
                        'duration_output': 'categorical_crossentropy'},
                  optimizer='adam')
    model.summary()
    return model

def train():
    # Create the folder to store models if it doesn't exist
    os.makedirs("model", exist_ok=True)
    sequence_length = 100
    notes = get_notes()
    (X_pitch, X_duration, y_pitch, y_duration, 
     pitch_to_int, int_to_pitch, duration_to_int, int_to_duration, 
     sequence_length, n_pitch, n_duration) = prepare_sequences(notes, sequence_length)
    
    # Check for existing checkpoint files
    checkpoint_files = glob.glob("model/weights-improvement-*.keras")
    if checkpoint_files:
        latest_checkpoint = sorted(checkpoint_files)[-1]
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model = load_model(latest_checkpoint)
        # Extract the epoch number from the filename.
        # For instance, if filename is "model/weights-improvement-98-0.1234.keras"
        initial_epoch = int(latest_checkpoint.split('-')[2])
    else:
        print("No checkpoint found, starting training from scratch.")
        model = create_network(sequence_length, n_pitch, n_duration)
        initial_epoch = 0
    
    # Define a checkpoint to save the best model (use '.keras' extension for new Keras versions)
    filepath = "model/weights-improvement-{epoch:02d}-{loss:.4f}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit([X_pitch, X_duration], [y_pitch, y_duration], initial_epoch=initial_epoch,
              epochs=100, batch_size=64, callbacks=callbacks_list)
    
    model.save("model/lstm_model.keras")
    print("Model saved as model/lstm_model.keras")
    
    # Save the mappings and parameters for later use in generation.
    with open("model/notes.pkl", "wb") as f:
        pickle.dump({
            "pitch_to_int": pitch_to_int,
            "int_to_pitch": int_to_pitch,
            "duration_to_int": duration_to_int,
            "int_to_duration": int_to_duration,
            "sequence_length": sequence_length,
            "n_pitch": n_pitch,
            "n_duration": n_duration
        }, f)
    print("Mapping and parameters saved as model/notes.pkl")

if __name__ == '__main__':
    train()
