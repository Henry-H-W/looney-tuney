from MinorMusicGenerator import MinorMusicGenerator
from music21 import converter, note, stream, chord, key, meter
import random
import collections
import os

def extend_midi(input_filepath: str, output_filepath: str, additional_intervals: int = 30):
    OCTAVE_SHIFT = 12
    BAR_DELAY = 12  # 3 bars delay (assuming 4/4 time)

    # Load the existing MIDI file
    original_stream = converter.parse(input_filepath)
    extended_stream = stream.Stream()

    # Copy original MIDI contents to the new stream
    for element in original_stream.flat.notes:
        extended_stream.append(element)

    # --- Step 1: Key & Scale Detection ---
    detected_key = original_stream.analyze('key')
    root_note = detected_key.tonic.midi

    if root_note < 59:
        root_note = 59
    elif root_note > 70:
        root_note = 70

    mode = detected_key.mode  # 'major' or 'minor'
    
    print(f"Detected Key: {detected_key}")

    # Initialize MinorMusicGenerator based on the detected key
    new_song_generator = MinorMusicGenerator(root_note)

    # --- Step 2: Extract Rhythmic Patterns ---
    durations = [n.quarterLength for n in original_stream.flat.notes]
    most_common_durations = collections.Counter(durations).most_common()
    top_durations = [d[0] for d in most_common_durations[:4]]  # Pick 4 most common note lengths

    print(f"Extracted Rhythmic Patterns: {top_durations}")

    # --- Step 3: Identify Common Chord Progressions ---
    chords = []
    for element in original_stream.flat.notes:
        if isinstance(element, chord.Chord):
            chords.append(element.pitches)
    
    most_common_chords = collections.Counter(chords).most_common()
    top_chords = [chord.Chord(c[0]) for c in most_common_chords[:4]]

    print(f"Common Chords Found: {top_chords}")

    # --- Step 4: Generate New Music Based on Extracted Features ---
    delayed_right_hand_notes = []

    def add_one_interval(current_index=0, right_hand_shift: int = 0,
                     current_velocity: int = 90, left_hand_shift: int = 0):
        # Generating notes for the right hand (delayed)
        current_index_for_the_right_hand = current_index + BAR_DELAY
        shift = right_hand_shift * OCTAVE_SHIFT

        # Choose rhythmic pattern based on input MIDI
        duration = random.choice(top_durations) if top_durations else 1.0

        # Right hand melody (delayed)
        if random.randint(0, 8) % 7 != 0:
            random_note = new_song_generator.correct_notes[random.randint(0, 6)] + shift
            my_note = note.Note(random_note, quarterLength=duration)
            my_note.volume.velocity = current_velocity
            delayed_right_hand_notes.append((current_index_for_the_right_hand, my_note))

        # Left hand harmony (starts immediately)
        sequence_of_notes = new_song_generator.baselines[random.randint(0, 2)]

        for note_i in range(0, 12):
            durations = [4, 0.5, 1, 1.5, 2]  # 16th, 8th, quarter, dotted quarter, half notes
            random_duration = random.choice(durations)
            if random.random() < 0.7:  # 70% chance to move stepwise
                cur_note = sequence_of_notes[(note_i + random.choice([-1, 0, 1])) % len(sequence_of_notes)]
            else:  # 30% chance to leap
                cur_note = sequence_of_notes[random.randint(0, len(sequence_of_notes) - 1)]
            if random.randint(0, 8) % 7 != 0:
                new_note = note.Note(cur_note, quarterLength=random_duration)
                new_note.volume.velocity = 70
                extended_stream.insert(current_index, new_note)
            current_index += 0.33

        # Occasionally insert common chords (FIX: Create a new copy of the chord)
        if random.random() < 0.3 and top_chords:
            chosen_chord = random.choice(top_chords)
            new_chord = chord.Chord(chosen_chord.pitches)  # Create a new chord instance
            new_chord.quarterLength = duration
            new_chord.volume.velocity = 80
            extended_stream.insert(current_index, new_chord)  # Insert fresh copy

    # Find the last note's timestamp from the original file
    last_offset = max(n.offset for n in original_stream.flat.notes)

    # Generate additional music based on detected key, rhythm, and harmony
    for i in range(additional_intervals):
        add_one_interval(current_index=last_offset + 4 * i,
                         right_hand_shift=random.randint(-1, 1),
                         current_velocity=random.randint(80, 110),
                         left_hand_shift=random.randint(-3, -1))

    # Insert delayed right-hand notes into the stream
    for offset, delayed_note in delayed_right_hand_notes:
        extended_stream.insert(offset, delayed_note)

    # Save the extended MIDI file
    extended_stream.write('mid', fp=output_filepath)


if __name__ == '__main__':
    # find the most recent MIDI file in the directory
    midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the directory.")

    # get the most recently created/modified MIDI file
    latest_midi = max(midi_files, key=os.path.getctime)
    extend_midi(latest_midi, 'extended_output.mid', additional_intervals=30)