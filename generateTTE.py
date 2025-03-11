from MinorMusicGenerator import MinorMusicGenerator
from music21 import note, stream, chord
import random

def generate_music(scale: int, filepath: str):
    OCTAVE_SHIFT = 12
    new_song_generator = MinorMusicGenerator(scale)
    myStream = stream.Stream()

    intervals = 30
    note_duration = [4, 2, 1, 0.66]
    number_of_notes = [2, 2, 8, 12]

    volumes = [100, 50, 60, 60, 70, 80, 100, 80, 70, 60, 50, 50]

    def add_one_interval(current_index=0, right_hand_shift: int = 0,
                         current_velocity: int = 90, left_hand_shift: int = 0):
        # generating notes for the right hand
        current_index_for_the_right_hand = current_index
        current_note_duration_index = random.randint(0, len(note_duration) - 1)
        current_number_of_notes = number_of_notes[current_note_duration_index]
        current_duration = note_duration[current_note_duration_index]
        shift: int = right_hand_shift * OCTAVE_SHIFT

        # generating the sequence of notes for the right hand
        for note_i in range(current_number_of_notes):
            if random.randint(0, 8) % 7 != 0:
                random_note = new_song_generator.correct_notes[random.randint(0, 6)] + shift
                my_note = note.Note(random_note, quarterLength=current_duration + 1)
                my_note.volume.velocity = current_velocity
                myStream.insert(current_index_for_the_right_hand, my_note)
            current_index_for_the_right_hand += current_duration

        # generating the sequence of notes for the left hand
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
                myStream.insert(current_index, new_note)
            current_index += 0.33

    for i in range(intervals):
        add_one_interval(current_index=4 * i,
                         right_hand_shift=random.randint(-1, 1),
                         current_velocity=random.randint(80, 110),
                         left_hand_shift=random.randint(-3, -1))
    add_one_interval(current_index=4 * intervals, current_velocity=50)
    myStream.write('mid', fp=filepath)


if __name__ == '__main__':
    random_scale = random.randint(48, 72)  # random root note within a musical range
    generate_music(64, 'generated_output.mid')