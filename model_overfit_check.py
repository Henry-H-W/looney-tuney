'''
this OPTIONAL helper function mathematically determines whether YOUR model is 
overfit or just a boss at composing 😎 it essentially works by taking 
your generated midi, pulling its note distributions (80% G#3, 54% B, 3% F#36, etc.)
and comparing them to the note distributions of each individual track 
from your dataset in a 'who-is-the-father' style test. using cosine similarity 
(mathematically proven to work 👍) if your midi is similar to a particular 
dataset track to the following degrees:
- similarity = 1.0 → exact copy (yup that's the father, high risk of overfitting)
- similarity > 0.8 → very similar (model may be memorizing dataset patterns)
- similarity ~ 0.5-0.7 → somewhat similar (good balance of structure & 
    variation, everyone's the father 👍)
- similarity < 0.4 → very different (model is generating "unique" music)
you can determine if your model has overfit! good luck ~~
'''

import numpy as np
import glob
import collections
from sklearn.metrics.pairwise import cosine_similarity
from music21 import converter, note, chord

def extract_notes_from_midi(midi_file):
    """ extract notes and chords from a MIDI file """
    try:
        midi = converter.parse(midi_file)
    except Exception as e:
        print(f"Error reading {midi_file}: {e}")
        return []

    notes = []
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(element.nameWithOctave)  # store note names with octaves (e.g., C4, G#5)
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(n.nameWithOctave for n in element.pitches))  # Convert chords to string format

    return notes

def compute_note_frequencies(midi_file):
    """ compute note frequency distribution for a single MIDI file """
    notes = extract_notes_from_midi(midi_file)
    return collections.Counter(notes)

def compute_cosine_similarity(counts1, counts2):
    """ compute cosine similarity between two note distributions """
    unique_notes = sorted(set(counts1.keys()).union(set(counts2.keys())))

    vector1 = np.array([counts1.get(note, 0) for note in unique_notes])
    vector2 = np.array([counts2.get(note, 0) for note in unique_notes])

    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0  # avoid division by zero

    vector1 = vector1 / np.linalg.norm(vector1)
    vector2 = vector2 / np.linalg.norm(vector2)

    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

# paths (update these)
dataset_folder = "dataset"  # folder containing MIDI dataset
generated_midi = "20e4.mid"  # path to generated MIDI file

# compute note distribution for the generated MIDI
generated_counts = compute_note_frequencies(generated_midi)

# loop through all dataset MIDI files and compare
similarity_results = []

for dataset_file in glob.glob(f"{dataset_folder}/*.mid"):
    dataset_counts = compute_note_frequencies(dataset_file)
    similarity_score = compute_cosine_similarity(dataset_counts, generated_counts)

    similarity_results.append((dataset_file, similarity_score))

# sort results from most to least similar
similarity_results.sort(key=lambda x: x[1], reverse=True)

# print results
print("\nsimilarity Scores (dataset vs generated MIDI):")
for file, score in similarity_results:
    print(f"{file}: {score:.3f}")
