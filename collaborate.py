# importing necessary libraries
import numpy as np  # for numerical operations (arrays, matrices, etc.)
import pandas as pd  # for handling structured data (e.g., CSV files, dataframes)
import matplotlib.pyplot as plt  # for data visualization (graphs, charts)

import os  # to handle file and directory operations
import glob  # to find files matching a pattern (e.g., all MIDI files in a folder)
import pickle  # to save and load serialized objects (like trained models or preprocessed data)
import datetime # for output file specification

# importing music21 - a python library for handling and analyzing music notation
from music21 import converter, instrument, stream, note, chord, pitch, interval
from collections import Counter
import random

# importing deep learning tools from Keras
from keras.models import Sequential  # for building sequential neural networks
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten  # different types of layers
from keras import utils  # utilities for handling labels, models, and training
from keras.callbacks import ModelCheckpoint  # to save the best model during training
from keras_self_attention import SeqSelfAttention  # self-attention mechanism for sequence models

# note: ensure that you have `keras_self_attention` installed before using it.
# you can install it using: pip install keras-self-attention

# running version 2.1.6 (assuming this is a specific requirement for compatibility)
# check your installed version with:
# import keras
# print(keras.__version__)

def generate():
    """ generate a piano midi file """

    # load the notes that were used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)  # load the saved notes data

    # get all unique pitch names (notes, chords, and rests) from the dataset
    pitchnames = sorted(set(item for item in notes))

    # get the total number of unique notes (vocabulary size)
    n_vocab = len(set(notes))

    # prepare the input sequences for generating new music
    network_input, normalized_input = prepare_sequences_output(notes, pitchnames, n_vocab)

    # create the model and load trained weights
    model = create_network_add_weights(normalized_input, n_vocab)

    # generate a sequence of new musical notes using the trained model
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)

    # convert the generated sequence of notes into a midi file
    create_midi(prediction_output)

def prepare_sequences_output(notes, pitchnames, n_vocab):
    """ prepare the sequences used by the neural network for generating music """

    # create a dictionary to map each unique note/chord/rest to an integer
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100  # define the length of each input sequence

    # initialize lists to store input sequences and corresponding outputs
    network_input = []
    output = []

    # create sequences of 100 notes each, using a sliding window approach
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]  # input sequence of 100 notes
        sequence_out = notes[i + sequence_length]  # the next note (prediction target)

        # convert input sequence notes to their integer representations
        network_input.append([note_to_int[char] for char in sequence_in])

        # convert the output note to its corresponding integer
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)  # number of training patterns (samples)

    # reshape the input into a 3d format required for lstm layers: (samples, time steps, features)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize the input values to range 0-1 to improve model performance
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)  # return raw input and normalized input

def create_network_add_weights(network_input, n_vocab):
    """ create the structure of the neural network and load pre-trained weights """

    model = Sequential()  # initialize a sequential model (linear stack of layers)

    # add a bidirectional lstm layer with 512 units
    # lstm processes the input sequences while bidirectional allows learning dependencies in both directions
    model.add(Bidirectional(LSTM(512, return_sequences=True),
                            input_shape=(network_input.shape[1], network_input.shape[2])))
    # input_shape must be specified in the first layer, using (time steps, features)

    # add a self-attention mechanism to help the model focus on important parts of the sequence
    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    # add dropout to prevent overfitting by randomly deactivating 30% of neurons
    model.add(Dropout(0.3))

    # add another lstm layer with 512 units, still returning sequences
    model.add(LSTM(512, return_sequences=True))

    # add another dropout layer to further reduce overfitting risk
    model.add(Dropout(0.3))

    # flatten the lstm output before passing it to dense layers (reshapes into a 1d vector)
    model.add(Flatten())

    # add a dense output layer with 'n_vocab' neurons (one per unique note/chord)
    model.add(Dense(n_vocab))

    # apply softmax activation to convert outputs into probabilities (multi-class classification)
    model.add(Activation('softmax'))

    # compile the model using categorical cross-entropy loss (suitable for multi-class problems)
    # rmsprop is used as the optimizer to improve training stability
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # load pre-trained weights to avoid training from scratch
    # this allows the model to generate music based on previously learned patterns

    # find the most recent MIDI file in the directory
    model_weights = [f for f in os.listdir() if f.endswith(".keras")]
    if not model_weights:
        raise FileNotFoundError("No model weights files found in the directory.")

    # get the most recently created/modified MIDI file
    latest_model_weights = max(model_weights, key=os.path.getctime)

    print(f"Processing most recent keras file: {latest_model_weights}")
    model.load_weights(latest_model_weights)

    return model  # return the model with loaded weights

def get_seed_note_from_latest_midi():
    """
    Find the most recent .mid file in the working directory, parse it,
    and return its last note as a string (e.g., "C3 1.0").
    If the last element is a chord, take the first note of the chord.
    """
    # Find all .mid files
    midi_files = glob.glob("*.mid")
    if not midi_files:
        raise FileNotFoundError("No .mid files found in the working directory.")
    
    # Get the most recent .mid file by creation/modification time
    latest_midi = max(midi_files, key=os.path.getctime)
    print(f"Using most recent midi file: {latest_midi} for seed note extraction")
    
    # Parse the MIDI file using music21
    midi_stream = converter.parse(latest_midi)
    
    # Try to partition by instrument; if unavailable, use flat notes
    parts = instrument.partitionByInstrument(midi_stream)
    if parts: 
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi_stream.flat.notes
    
    # Gather all notes and chords
    elements = []
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            # Format as "Pitch Duration"
            elements.append(f"{element.pitch} {element.duration.quarterLength}")
        elif isinstance(element, chord.Chord):
            # For chords, take the first pitch
            elements.append(f"{element.pitches[0]} {element.duration.quarterLength}")
    
    if not elements:
        raise ValueError("No note elements found in the latest MIDI file.")
    
    # Return the last element found
    seed_note = elements[-1]
    print("Extracted seed note:", seed_note)
    return seed_note

def next_seed(seed_str):
    """
    Given a seed note string (e.g., "C3 1.0"), return the next note by transposing
    the pitch up by one semitone while keeping the duration unchanged.
    """
    parts = seed_str.split()
    note_part = parts[0]
    duration = parts[1] if len(parts) > 1 else "1.0"
    p = pitch.Pitch(note_part)
    new_pitch = p.transpose(interval.Interval(1))  # transpose up by one semitone
    return f"{new_pitch.nameWithOctave} {duration}"

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ generate notes from the neural network based on a sequence of notes """
    
    # Create mappings from note names to integers and vice versa
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for number, note in enumerate(pitchnames)}
    
    # Use an initial seed note (without hardcoding fallback alternatives)
    seed_note = get_seed_note_from_latest_midi()  # starting point; you can change this as desired
    max_attempts = 12  # maximum fallback attempts (e.g., an octave's worth)
    pattern = None
    used_seed_note = None
    attempt = 0

    while attempt < max_attempts and pattern is None:
        if seed_note not in note_to_int:
            print(f"Seed note {seed_note} not found in training data. Trying next note.")
            seed_note = next_seed(seed_note)
            attempt += 1
            continue
        
        desired_seed_value = note_to_int[seed_note]
        # Instead of breaking at the first match, collect all matching sequences.
        matching_sequences = [seq.copy() for seq in network_input if seq[-1] == desired_seed_value]
        
        if matching_sequences:
            # Randomly pick one sequence from the list.
            pattern = random.choice(matching_sequences)
            used_seed_note = seed_note
            break
        else:
            print(f"No sequence ending with {seed_note} found. Trying next note.")
            seed_note = next_seed(seed_note)
            attempt += 1

    if pattern is None:
        raise ValueError("No seed sequence ending with a valid note candidate was found in network_input.")

    print("Selected seed sequence (notes):", [int_to_note[i] for i in pattern])

    prediction_output = []
    # Generate 150 notes using the seed sequence
    for note_index in range(150):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:]
    
    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file """
    offset = 0  # keeps track of time to avoid overlapping notes
    output_notes = []  # list to store the generated musical elements (notes, chords, rests)

    # iterate through each predicted pattern (note, chord, or rest)
    for pattern in prediction_output:
        pattern = pattern.split()  # split to separate the musical element and duration
        temp = pattern[0]         # musical element (note, chord, or rest)
        duration = pattern[1]     # duration as a string
        pattern = temp            # reassign the musical element back to pattern

        # check if the pattern represents a chord (multiple notes played together)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')  # split the chord into individual notes
            notes = []  # list to store note objects
            for current_note in notes_in_chord:
                if current_note.isdigit():
                    new_note = note.Note(int(current_note))
                else:
                    new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # check if the pattern represents a rest
        elif 'rest' in pattern:
            new_rest = note.Rest()  # create a rest object
            # Convert duration to float and cap it at 4.0 if necessary
            rest_duration = convert_to_float(duration)
            if rest_duration > 4.0:
                rest_duration = 4.0
            new_rest.duration.quarterLength = rest_duration
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            output_notes.append(new_rest)

        # if the pattern is a single note
        else:
            new_note = note.Note(pattern)  # create a note object
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase the offset by the (possibly capped) duration
        offset += (rest_duration if 'rest' in pattern else convert_to_float(duration))

    # create a midi stream from the generated notes and chords
    midi_stream = stream.Stream(output_notes)

    # write the midi stream to a file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    midi_filename = f"generated_music_{timestamp}.mid"
    midi_stream.write('midi', fp=midi_filename)
    print(f"Generated MIDI saved as {midi_filename}")


# helper function to convert fraction strings to float values
# yes i know convert_duration and convert_to_float can be merged, i'm just too lazy to do it
def convert_to_float(frac_str):
    try:
        return float(frac_str)  # try to directly convert the string to a float
    except ValueError:  # handle cases where the string is a fraction (e.g., "3/4")
        num, denom = frac_str.split('/')  # split numerator and denominator
        try:
            leading, num = num.split(' ')  # check for mixed fractions (e.g., "1 3/4")
            whole = float(leading)  # extract the whole number part
        except ValueError:
            whole = 0  # if no whole number part, set to zero
        frac = float(num) / float(denom)  # compute the fractional value
        return whole - frac if whole < 0 else whole + frac  # return the final float value

# run the generator to create a new midi file
generate()