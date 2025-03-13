from MinorMusicGenerator import MinorMusicGenerator
from music21 import converter, note, stream, chord
import random
import collections
import os

def extend_midi(input_filepath: str, output_filepath: str, additional_intervals: int = 15):
    OCTAVE_SHIFT = 12
    BAR_DELAY = 4  # 3 bars delay (assuming 4/4 time)

    # Load the existing MIDI file
    original_stream = converter.parse(input_filepath)
    extended_stream = stream.Stream()

    # Copy original MIDI contents to the new stream
    extended_stream = original_stream.flat

    # --- Step 1: Key & Scale Detection ---
    detected_key = original_stream.analyze('key')
    root_note = detected_key.tonic.midi
    mode = detected_key.mode  # 'major' or 'minor'

    # Constrain the root note to a reasonable range
    if root_note < 59:
        root_note = 59
    elif root_note > 70:
        root_note = 70

    print(f"Detected Key: {detected_key} ({mode})")

    # Initialize MinorMusicGenerator based on the detected key
    new_song_generator = MinorMusicGenerator(root_note)

    # --- Step 2: Extract Rhythmic Patterns ---
    durations = [n.quarterLength for n in original_stream.flat.notes]
    most_common_durations = collections.Counter(durations).most_common()
    top_durations = [d[0] for d in most_common_durations[:4]] if most_common_durations else [1.0]

    print(f"Extracted Rhythmic Patterns: {top_durations}")

    # --- Step 3: Use MinorMusicGenerator Chords ---
    minor_chords = new_song_generator.minor_chords
    additional_chords = new_song_generator.additional_chords
    generated_chords = minor_chords + additional_chords  # Combine all chords

    print(f"Generated Chords from Key: {generated_chords}")

    delayed_right_hand_notes = []

    def add_one_interval(current_index=0, right_hand_shift: int = 0,
                         current_velocity: int = 90, left_hand_shift: int = 0):
        """ Generates a musical interval based on extracted features and extends the piece. """

        # Right-hand melody (delayed)
        current_index_for_the_right_hand = current_index + BAR_DELAY
        shift = right_hand_shift * OCTAVE_SHIFT
        duration = random.choice(top_durations)

        if random.random() < 0.8:  # 80% chance to play a melody note
            random_note = new_song_generator.correct_notes[random.randint(0, 6)] + shift
            my_note = note.Note(random_note, quarterLength=duration)
            my_note.volume.velocity = current_velocity
            delayed_right_hand_notes.append((current_index_for_the_right_hand, my_note))

        # Left-hand harmony (chordal support and bass movement)
        sequence_of_notes = new_song_generator.baselines[random.randint(0, 2)]

        # Ensure chords play **every bar**
        if random.random() < 0.7:
            # Select a chord from the minor_chords list
            chosen_chord = random.choice(new_song_generator.minor_chords)
            # Use only the first three notes for a simple triad
            simple_chord = chosen_chord[:3]
            new_chord = chord.Chord(simple_chord)
            new_chord.quarterLength = max(4, duration * 2)  # At least one bar long
            new_chord.volume.velocity = 80
            extended_stream.insert(current_index, new_chord)

        # Generate bassline (harmony)
        for _ in range(8):  # Play 4 bass notes per measure
            random_duration = random.choice([0.5, 1, 1.5, 2])  # Quarter, dotted quarter, half
            cur_note = sequence_of_notes[random.randint(0, len(sequence_of_notes) - 1)]

            new_note = note.Note(cur_note, quarterLength=random_duration)
            new_note.volume.velocity = 70
            extended_stream.insert(current_index, new_note)

            current_index += random_duration  # Move forward in time

    # Find the last note's timestamp in the original file
    last_offset = max(n.offset for n in original_stream.flat.notes)
    print(last_offset)

    # Generate additional music based on detected key, rhythm, and harmony
    for i in range(additional_intervals):
        add_one_interval(
            current_index=last_offset + 4 * i,
            right_hand_shift=random.randint(-1, 1),
            current_velocity=random.randint(80, 110),
            left_hand_shift=random.randint(-3, -1),
        )

    # Insert delayed right-hand notes into the stream
    for offset, delayed_note in delayed_right_hand_notes:
        extended_stream.insert(offset, delayed_note)

    # Save the extended MIDI file
    extended_stream.write('mid', fp=output_filepath)


if __name__ == '__main__':
    # Find the most recent MIDI file in the directory
    midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the directory.")

    # Get the most recently created/modified MIDI file
    latest_midi = max(midi_files, key=os.path.getctime)
    print(f"Extending MIDI file: {latest_midi}")

    extend_midi(latest_midi, 'extended_output.mid', additional_intervals=15)