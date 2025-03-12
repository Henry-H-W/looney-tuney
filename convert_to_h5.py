import tensorflow as tf
from keras.models import Sequential  # for building sequential neural networks
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten  # different types of layers
from keras_self_attention import SeqSelfAttention  # self-attention mechanism for sequence models
import pickle  # to save and load serialized objects (like trained models or preprocessed data)

# load the notes that were used to train the model
with open('data/notes', 'rb') as filepath:
    notes = pickle.load(filepath)  # load the saved notes data

# get the total number of unique notes (vocabulary size)
n_vocab = len(set(notes))

model = Sequential()  # initialize a sequential model (linear stack of layers)

# add a bidirectional lstm layer with 512 units
# lstm processes the input sequences while bidirectional allows learning dependencies in both directions
model.add(Bidirectional(LSTM(512, return_sequences=True),
                        input_shape=(100, 1)))
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


# Load the .keras model
model.load_weights("weights-epoch035-0.2775.keras")

# Save it in .h5 format
model.save("weights_epoch035.h5")