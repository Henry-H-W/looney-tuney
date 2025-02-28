# importing necessary libraries
import numpy as np  # for numerical operations (arrays, matrices, etc.)
import pandas as pd  # for handling structured data (e.g., CSV files, dataframes)
import matplotlib.pyplot as plt  # for data visualization (graphs, charts)

import os  # to handle file and directory operations
import glob  # to find files matching a pattern (e.g., all MIDI files in a folder)
import pickle  # to save and load serialized objects (like trained models or preprocessed data)
import datetime # for output file specification

# importing music21 - a python library for handling and analyzing music notation
from music21 import converter, instrument, stream, note, chord
from collections import Counter

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

def train_network(notes, n_vocab):
    """
    train a Neural Network to generate music based on a given sequence of notes

    parameters:
    notes (list): a list of musical notes and chords extracted from MIDI files
    n_vocab (int): the number of unique notes/chords in the dataset (vocabulary size)

    this function follows three main steps:
    1. prepare the input sequences and corresponding output targets for the model
    2. create the LSTM-based neural network architecture
    3. train the model using the prepared sequences
    """

    # step 1: convert the notes into a format that the neural network can understand
    network_input, network_output = prepare_sequences(notes, n_vocab)

    print(f"Total training sequences generated: {len(network_input)}")

    # step 2: create the LSTM-based neural network model
    model = create_network(network_input, n_vocab)

    # step 3: train the model using the prepared input and output sequences
    train(model, network_input, network_output)

def convert_duration(duration_value):
    """ Converts duration to float, handling fractions like '1/3'. """
    try:
        return float(duration_value)
    except ValueError:
        return float(Fraction(duration_value))  # Convert fraction (e.g., "1/3") to decimal

def adjust_octave(pitch_name, shift=-1):
    """ Adjusts the octave of a note to fix incorrect octave shifting. """
    if pitch_name[-1].isdigit():  # Ensure last character is an octave number
        note_part = pitch_name[:-1]  # Get note name (e.g., 'B')
        octave_part = int(pitch_name[-1])  # Get octave number
        return f"{note_part}{octave_part + shift}"  # Apply shift
    return pitch_name  # If no octave detected, return as is

def get_notes():
    """ Extracts all notes, chords, and rests from MIDI files. """

    # check if the "data/notes" file already exists to avoid unnecessary re-parsing
    if os.path.exists('data/notes'):
        print("skipping midi parsing - 'data/notes' already exists.")
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)  # load previously parsed notes from the saved file
        return notes  # return the existing notes data

    notes = []  # Store cleaned notes, chords, and rests
    last_offset = 0.0  # Keep track of the last note's offset

    # locate all MIDI files in the dataset directory (if you have .midi files instead of .mid, edit the line below)
    midi_files = glob.glob("dataset/*.mid")
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in 'dataset/' directory.")

    # iterate through all MIDI files in the dataset
    for file in midi_files:
        try:
            midi = converter.parse(file)  # load MIDI file
            print(f"Parsing {file} ...")

            # try to extract instruments, otherwise flatten
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() if s2 else midi.flat.notes
            except:
                notes_to_parse = midi.flat.notes

            # iterate through each musical element and store it (note, chord, or rest)
            for element in notes_to_parse:
                duration_value = convert_duration(element.quarterLength)  # ensure duration is a float
                # possible experiment: work with time (seconds) instead of note durations for absolute precision

                # only add rests if there is an actual gap (this was a problem for some reason)
                if element.offset > last_offset:
                    rest_duration = element.offset - last_offset
                    if rest_duration >= 0.25: 
                        notes.append(f"rest {rest_duration}")

                if isinstance(element, note.Note):  # 🎵 Single Note
                    fixed_note = adjust_octave(element.nameWithOctave, shift=-1)
                    notes.append(f"{fixed_note} {duration_value}")

                elif isinstance(element, chord.Chord):  # 🎶 Chord
                    chord_notes = ".".join(adjust_octave(n.nameWithOctave, shift=-1) for n in element.pitches)
                    notes.append(f"{chord_notes} {duration_value}")

                # update last_offset to track the most recent note's position
                last_offset = element.offset + duration_value

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue  # continue processing other files

    # apply hard cutoff: keep only the 15000 most common notes to avoid resource exhaustion errors
    note_counts = Counter(notes)
    most_common_notes = {note for note, _ in note_counts.most_common(15000)}

    # remove rare notes entirely
    filtered_notes = [note for note in notes if note in most_common_notes]

    # save cleaned notes to a pickle file for future use
    os.makedirs('data', exist_ok=True)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(filtered_notes, filepath)

    print(f"Successfully extracted {len(filtered_notes)} elements from {len(midi_files)} MIDI files.")
    return filtered_notes

def prepare_sequences(notes, n_vocab):
    """ prepare the sequences used by the neural network """

    sequence_length = 100  # define the length of each input sequence

    # get all unique pitch names (notes, chords, and rests) and sort them
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary that maps each unique note/chord/rest to an integer
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # initialize lists to store input sequences and corresponding target outputs
    network_input = []
    network_output = []

    # create input sequences and their corresponding output notes
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]  # take a sequence of 100 notes as input
        sequence_out = notes[i + sequence_length]  # the next note after the sequence is the target output

        # convert notes in the sequence to their corresponding integer values
        network_input.append([note_to_int[char] for char in sequence_in])
        # convert the target output note to its integer representation
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)  # number of training samples (patterns)

    # reshape the input into a 3D format required for lstm layers: (samples, time steps, features)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize the input values to a range of 0 to 1 (helps lstm training)
    network_input = network_input / float(n_vocab)

    # convert the output values into a one-hot encoded format
    network_output = utils.to_categorical(network_output)

    return (network_input, network_output)  # return the processed input and output sequences

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """

    model = Sequential()  # initialize a sequential model (a linear stack of layers)

    # add a bidirectional lstm layer with 512 units
    model.add(Bidirectional(LSTM(512,
        input_shape=(network_input.shape[1], network_input.shape[2]),  # shape: (time steps, features)
        return_sequences=True)))  # return sequences to allow stacking more lstm layers

    # add a self-attention layer to help the model focus on important time steps in the sequence
    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    # add dropout to prevent overfitting (randomly deactivates 30% of neurons)
    model.add(Dropout(0.3))

    # add another lstm layer with 512 units, still returning sequences
    model.add(LSTM(512, return_sequences=True))

    # add another dropout layer to further reduce overfitting risk
    model.add(Dropout(0.3))

    # flatten the output before passing it to dense layers (reshapes it into a 1d vector)
    model.add(Flatten())  # ensures compatibility with the dense output layer

    # add a dense output layer with 'n_vocab' neurons (one per unique note/chord)
    model.add(Dense(n_vocab))

    # apply softmax activation to convert outputs into probabilities (multi-class classification)
    model.add(Activation('softmax'))

    # compile the model using categorical cross-entropy loss and rmsprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # (maybe experiment with different optimizers?)

    return model  # return the compiled model


def train(model, network_input, network_output):
    """ train the neural network """

    # let's set up a model checkpoint system that saves the model weights after every 5 epochs of training!
    batch_size = 64
    steps_per_epoch = len(network_input) // batch_size 
    save_freq = steps_per_epoch * 5

    # Use the custom callback
    filepath = os.path.abspath("weights-epoch{epoch:03d}-{loss:.4f}.keras")
    checkpoint = ModelCheckpoint(
        filepath,
        save_freq=save_freq, #Every 10 epochs
        monitor='loss',
        verbose=1,
        save_best_only=False,
        mode='min'
    )

    # Then pass the callback to model.fit()
    model.fit(network_input, network_output,
              epochs=5,
              batch_size=64,
              callbacks=[checkpoint]
    )

# load all musical notes, chords, and rests from midi files
notes = get_notes()

# get the total number of unique pitch names (distinct notes, chords, and rests)
n_vocab = len(set(notes))  # converts list to set to remove duplicates, then gets its length

print(f"Vocabulary size (n_vocab): {n_vocab}")

# train the model using the extracted notes and the vocabulary size
# note: before running the model, make sure you have access to a GPU!
train_network(notes, n_vocab) # comment if you already have a weights file you want to use