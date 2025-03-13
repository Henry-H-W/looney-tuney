from MinorMusicGenerator import MinorMusicGenerator
import music21
from music21 import note, stream, chord
import pretty_midi
import random
import collections
import os

def pm_to_stream(pm, beat_duration):
    """
    Convert a pretty_midi.PrettyMIDI object into a music21 stream.
    All instruments are merged into one part. Note offsets and durations
    are calculated in quarter lengths using the given beat_duration.
    """
    s = stream.Score()
    part = stream.Part()
    
    # Group notes by offset so that simultaneous notes can form chords
    notes_by_offset = {}
    for instrument in pm.instruments:
        for n in instrument.notes:
            offset = n.start / beat_duration  # convert time (sec) to quarterLength
            duration = (n.end - n.start) / beat_duration
            m21_note = note.Note(n.pitch)
            m21_note.quarterLength = duration
            m21_note.offset = offset
            m21_note.volume.velocity = n.velocity
            notes_by_offset.setdefault(offset, []).append(m21_note)
    
    # Insert notes or chords into the part based on offset
    for offset in sorted(notes_by_offset.keys()):
        notes_list = notes_by_offset[offset]
        if len(notes_list) == 1:
            part.insert(offset, notes_list[0])
        else:
            # Create a chord from simultaneous notes
            pitches = [n.pitch for n in notes_list]
            chord_duration = max(n.quarterLength for n in notes_list)
            m21_chord = chord.Chord(pitches)
            m21_chord.quarterLength = chord_duration
            m21_chord.offset = offset
            avg_velocity = sum(n.volume.velocity for n in notes_list) // len(notes_list)
            m21_chord.volume.velocity = avg_velocity
            part.insert(offset, m21_chord)
    
    s.insert(0, part)
    s.makeMeasures(inPlace=True)
    return s

def extend_midi(input_filepath: str, output_filepath: str, additional_intervals: int = 15):
    OCTAVE_SHIFT = 12
    BAR_DELAY = 4  # assuming 4 beats delay (i.e. one bar in 4/4)
    
    # --- Load MIDI with pretty_midi and compute beat_duration ---
    pm = pretty_midi.PrettyMIDI(input_filepath)
    tempos, _ = pm.get_tempo_changes()
    if tempos.size == 0 or tempos[0] == 0:
        tempo = 120  # default tempo if missing
    else:
        tempo = tempos[0]
    beat_duration = 60.0 / tempo  # seconds per beat
    
    # Convert pretty_midi to a music21 stream
    original_stream = pm_to_stream(pm, beat_duration)
    
    # For debugging, write out the converted original stream
    # original_stream.write('mid', fp="og_" + output_filepath)
    extended_stream = original_stream.flat
    # extended_stream.write('mid', fp="test_" + output_filepath)
    
    # --- Step 1: Key & Scale Detection ---
    try:
        detected_key = original_stream.flatten().analyze('key')
    except music21.analysis.discrete.DiscreteAnalysisException:
        print("Key analysis failed, defaulting to C major")
        detected_key = music21.key.Key('C')
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
        """
        New version of add_one_interval based on generateTTE.py.
        This replaces the original right-hand and left-hand generation code,
        while using collaborateTTE's extended_stream and delayed_right_hand_notes.
        """
        # Parameters for the right-hand generation (from generateTTE)
        note_duration = [4, 2, 1, 0.66]
        number_of_notes = [2, 2, 8, 12]
        
        # Right-hand melody (with a delay as in collaborateTTE)
        current_index_for_the_right_hand = current_index + BAR_DELAY
        current_note_duration_index = random.randint(0, len(note_duration) - 1)
        current_number_of_notes = number_of_notes[current_note_duration_index]
        current_duration = note_duration[current_note_duration_index]
        shift = right_hand_shift * OCTAVE_SHIFT

        for note_i in range(current_number_of_notes):
            if random.randint(0, 8) % 7 != 0:
                # Select a note from the full range of correct_notes
                random_note = new_song_generator.correct_notes[random.randint(0, len(new_song_generator.correct_notes) - 1)] + shift
                my_note = note.Note(random_note, quarterLength=current_duration + 1)
                my_note.volume.velocity = current_velocity
                delayed_right_hand_notes.append((current_index_for_the_right_hand, my_note))
            current_index_for_the_right_hand += current_duration

        # Left-hand generation (adopted from generateTTE)
        sequence_of_notes = new_song_generator.baselines[random.randint(0, len(new_song_generator.baselines) - 1)]
        for note_i in range(12):
            durations = [4, 0.5, 1, 1.5, 2]  # possible note durations
            random_duration = random.choice(durations)
            if random.random() < 0.7:  # 70% chance for stepwise motion
                cur_note = sequence_of_notes[(note_i + random.choice([-1, 0, 1])) % len(sequence_of_notes)]
            else:  # 30% chance for a leap
                cur_note = sequence_of_notes[random.randint(0, len(sequence_of_notes) - 1)]
            if random.randint(0, 8) % 7 != 0:
                new_note = note.Note(cur_note, quarterLength=random_duration)
                new_note.volume.velocity = 70
                extended_stream.insert(current_index, new_note)
            current_index += 0.33  # small time step increment for left-hand notes

    
    # Find the last note's timestamp in the original stream
    last_offset = max(n.offset for n in original_stream.flat.notes)
    print("Last offset in original piece:", last_offset)
    
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
    print("Extended MIDI saved to", output_filepath)


if __name__ == '__main__':
    midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the directory.")
    
    latest_midi = max(midi_files, key=os.path.getctime)
    print(f"Extending MIDI file: {latest_midi}")
    
    extend_midi(latest_midi, 'extended_output.mid', additional_intervals=15)