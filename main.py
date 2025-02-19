# importing necessary libraries
import numpy as np  # for numerical operations (arrays, matrices, etc.)
import pandas as pd  # for handling structured data (e.g., CSV files, dataframes)
import matplotlib.pyplot as plt  # for data visualization (graphs, charts)

import os  # to handle file and directory operations
import glob  # to find files matching a pattern (e.g., all MIDI files in a folder)
import pickle  # to save and load serialized objects (like trained models or preprocessed data)

# importing music21 - a python library for handling and analyzing music notation
from music21 import converter, instrument, stream, note, chord

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
    """
    Extracts **all** notes, chords, and **real** rests from MIDI files.

    Fixes:
    - **Ensures rests are only added when there's an actual gap**.
    - **Corrects octave shifts** (down by 1).
    - **Preserves exact note timing from the MIDI file**.
    """

    # check if the "data/notes" file already exists to avoid unnecessary re-parsing
    if os.path.exists('data/notes'):
        print("skipping midi parsing - 'data/notes' already exists.")
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)  # load previously parsed notes from the saved file
        return notes  # return the existing notes data

    notes = []  # Store cleaned notes, chords, and rests
    last_offset = 0.0  # Keep track of the last note's offset

    midi_files = glob.glob("dataset/*.mid")
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in 'dataset/' directory.")

    for file in midi_files:
        try:
            midi = converter.parse(file)  # Load MIDI file
            print(f"Parsing {file} ...")

            # Try to extract instruments, otherwise flatten
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse() if s2 else midi.flat.notes
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                duration_value = convert_duration(element.quarterLength)  # Ensure duration is a float

                if element.offset > last_offset:
                    rest_duration = element.offset - last_offset
                    if rest_duration >= 0.25:  # Avoid micro-rests
                        notes.append(f"rest {rest_duration}")

                if isinstance(element, note.Note):  # 🎵 Single Note
                    fixed_note = adjust_octave(element.nameWithOctave, shift=-1)
                    notes.append(f"{fixed_note} {duration_value}")

                elif isinstance(element, chord.Chord):  # 🎶 Chord
                    chord_notes = ".".join(adjust_octave(n.nameWithOctave, shift=-1) for n in element.pitches)
                    notes.append(f"{chord_notes} {duration_value}")

                last_offset = element.offset + duration_value

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save cleaned notes to a pickle file
    os.makedirs('data', exist_ok=True)
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    print(f"Successfully extracted {len(notes)} elements from {len(midi_files)} MIDI files.")
    return notes

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

    return model  # return the compiled model


def train(model, network_input, network_output):
    """ train the neural network """

    batch_size = 64
    steps_per_epoch = len(network_input) // batch_size  # or use math.ceil(...)
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
              epochs=30,
              batch_size=64,
              callbacks=[checkpoint]
    )

# load all musical notes, chords, and rests from midi files
notes = get_notes()

# get the total number of unique pitch names (distinct notes, chords, and rests)
n_vocab = len(set(notes))  # converts list to set to remove duplicates, then gets its length

# train the model using the extracted notes and the vocabulary size
# note: before running the model, make sure you have access to a GPU!
#train_network(notes, n_vocab)

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
    model.load_weights('weights-epoch005-5.6513.keras')

    return model  # return the model with loaded weights

def generate_notes(model, network_input, pitchnames, n_vocab):
    """ generate notes from the neural network based on a sequence of notes """

    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)

    # create a dictionary to map integer values back to their corresponding notes
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # get the starting sequence pattern from the input data
    pattern = network_input[start]

    # initialize an empty list to store the generated notes
    prediction_output = []

    # generate 500 notes (this controls the length of the generated music)
    for note_index in range(500):
        # reshape the pattern to match the lstm model's expected input shape: (samples, time steps, features)
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))

        # normalize the input values to match the training scale
        prediction_input = prediction_input / float(n_vocab)

        # get the model's prediction for the next note
        prediction = model.predict(prediction_input, verbose=0)

        # get the index of the highest probability note from the prediction output
        index = np.argmax(prediction)

        # convert the predicted index back to its corresponding note
        result = int_to_note[index]

        # store the predicted note
        prediction_output.append(result)

        # update the input pattern by appending the new prediction and removing the first element
        pattern.append(index)
        pattern = pattern[1:len(pattern)]  # keep the sequence length constant

    return prediction_output  # return the list of generated notes

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file """

    offset = 0  # keeps track of time to avoid overlapping notes
    output_notes = []  # list to store the generated musical elements (notes, chords, rests)

    # iterate through each predicted pattern (note, chord, or rest)
    for pattern in prediction_output:
        pattern = pattern.split()  # split the pattern to separate the note/chord name and duration
        temp = pattern[0]  # extract the musical element (note, chord, or rest)
        duration = pattern[1]  # extract the duration of the note/chord/rest
        pattern = temp  # assign the extracted note/chord/rest back to pattern

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

        # check if the pattern represents a rest (a pause in the music)
        elif 'rest' in pattern:
            new_rest = note.Rest()  # create a rest without passing "rest" as an argument
            new_rest.duration.quarterLength = convert_to_float(duration)  # set the duration explicitly
            new_rest.offset = offset  # set the timing offset
            new_rest.storedInstrument = instrument.Piano()  # assign the instrument to piano
            output_notes.append(new_rest)  # add the rest to the output

        # if the pattern is a single note
        else:
            new_note = note.Note(pattern)  # create a note object
            new_note.offset = offset  # set the timing offset
            new_note.storedInstrument = instrument.Piano()  # assign the instrument to piano
            output_notes.append(new_note)  # add the note to the output

        # increase the offset to space out the notes and prevent stacking
        offset += convert_to_float(duration)

    # create a midi stream from the generated notes and chords
    midi_stream = stream.Stream(output_notes)

    # write the midi stream to a file
    midi_stream.write('midi', fp='test_output.mid')


# helper function to convert fraction strings to float values
# source: https://stackoverflow.com/questions/1806278/convert-fraction-to-float
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