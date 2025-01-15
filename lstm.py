from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.utils import to_categorical
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from music21 import converter, note, chord
import os
import pickle
import numpy as np

# Allocate about 80% of free memory
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data_folder = 'data'
files = [os.path.join(data_folder, file) for file
         in os.listdir(data_folder) if file.endswith('.mid')]

def get_notes_from_the_files(files):
    """ Get all the notes and chords from the MIDI files in the specified directory """
    notes = []
    for file in files:
        print(f"{file} parsing...")
        midi = converter.parse(file)
        notes_to_parse = midi.flatten()
        # D C E -> D.C.E
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.pitches))
    return notes

notes_file_path = f'{data_folder}/notes'

if os.path.exists(notes_file_path):
   # Loading notes from the file
   with open('data/notes', 'rb') as filepath:
       notes = pickle.load(filepath)
else:
   # creating a list of notes
   notes = get_notes_from_the_files(files)
   # saving notes to the file:
   with open('data/notes', 'wb') as filepath:
       pickle.dump(notes, filepath)

SEQUENCE_LENGTH = 100

# Create input sequences and corresponding output
unique_notes = sorted(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
int_to_note = dict((number, note) for number, note in enumerate(unique_notes))

input_sequences = []
output_sequences = []

for i in range(len(notes) - SEQUENCE_LENGTH):
    sequence_in = notes[i:i + SEQUENCE_LENGTH]
    sequence_out = notes[i + SEQUENCE_LENGTH]
    input_sequences.append([note_to_int[char] for char in sequence_in])
    output_sequences.append(note_to_int[sequence_out])

EPOCHS = 500  # Adjust as needed
BATCH_SIZE = 256  # Adjust as needed

# Reshape input sequences
X = np.reshape(input_sequences, (len(input_sequences), SEQUENCE_LENGTH, 1)).astype(np.float32) / float(len(unique_notes))

from scipy.sparse import csr_matrix

# One-hot encode output sequences
y = to_categorical(output_sequences)
y_sparse = csr_matrix(y)

model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

check_point = ModelCheckpoint('model/best_model.keras', save_best_only=True, monitor='loss')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Train the model
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[check_point])