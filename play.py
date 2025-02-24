import os
from pypianoroll import read, to_pretty_midi
from pydub import AudioSegment, playback
import soundfile as sf
import numpy as np
import pretty_midi

# define sample rate for better audio quality
sample_rate = 44100

# find the most recent midi file in the current directory
midi_files = [f for f in os.listdir() if f.endswith(".mid")]
if not midi_files:
    raise FileNotFoundError("no midi files found in the directory.")

# get the most recently created or modified midi file
latest_midi = max(midi_files, key=os.path.getctime)
original_file_name = os.path.splitext(latest_midi)[0]  # remove .mid extension

print(f"processing most recent midi file: {latest_midi}")

# load the midi file into a multitrack object
multitrack = read(latest_midi)

# convert the multitrack midi to pretty_midi format
pm = to_pretty_midi(multitrack)

# create a new pretty_midi object to store processed notes
new_pm = pretty_midi.PrettyMIDI()

# define legato overlap duration (20 milliseconds)
legato_overlap = 0.02

# process each instrument in the midi file
for instrument in pm.instruments:
    new_instrument = pretty_midi.Instrument(program=instrument.program)
    
    # sort notes by start time to ensure proper sequencing
    instrument.notes.sort(key=lambda note: note.start)
    
    last_end_time = 0  # keep track of the last note's end time
    i = 0  # initialize note index

    while i < len(instrument.notes):
        chord_notes = []  # store notes that start at the same time
        current_note = instrument.notes[i]

        # group notes that start at the same time (detect chords)
        while i < len(instrument.notes) and np.isclose(instrument.notes[i].start, current_note.start, atol=0.001):
            chord_notes.append(instrument.notes[i])
            i += 1

        # determine when the next note or chord starts
        next_start_time = instrument.notes[i].start if i < len(instrument.notes) else None

        # apply legato effect by extending each note's duration
        for note in chord_notes:
            new_start = max(0, note.start - legato_overlap)  # slightly move start time back
            if next_start_time:
                new_end = next_start_time - legato_overlap  # ensure overlap with the next note
            else:
                new_end = note.end  # keep original end time if it's the last note

            # ensure a minimum duration for each note
            new_end = max(new_start + 0.05, new_end)

            # create a new note with the adjusted timing
            new_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)

    new_pm.instruments.append(new_instrument)

# now new_pm contains the legato-processed midi

# create another pretty_midi object for additional effects
processed_pm = pretty_midi.PrettyMIDI()
for instrument in new_pm.instruments:
    processed_instrument = pretty_midi.Instrument(program=instrument.program)
    
    # sort notes by start time again
    instrument.notes.sort(key=lambda note: note.start)

    last_start_time = -1
    chord_notes = []

    for note in instrument.notes:
        new_velocity = 20  # set all notes to a soft velocity (ppp)

        # introduce slight timing variation (-10ms to +10ms) for a more human feel
        time_variation = np.random.uniform(-0.01, 0.01)
        new_start = max(0, note.start + time_variation)
        new_end = max(new_start + 0.05, note.end + time_variation)

        # slightly delay notes in long chords (chords lasting ≥1 second) to create an arpeggio effect
        if chord_notes and all(n.end - n.start >= 1.0 for n in chord_notes):
            for j, chord_note in enumerate(chord_notes):
                time_offset = j * np.random.uniform(0.02, 0.04)  # add small delays between notes
                chord_note.start += time_offset
                chord_note.end = max(chord_note.start + 0.05, chord_note.end + time_offset)

        # create the new processed note
        new_note = pretty_midi.Note(
            velocity=20,  # keep dynamics soft, 0 for softest and 127 for hardest
            pitch=note.pitch,
            start=new_start,
            end=new_end
        )
        processed_instrument.notes.append(new_note)

    processed_pm.instruments.append(processed_instrument)

# save the final processed midi file
edited_midi_name = f"edited_{original_file_name}.mid"
processed_pm.write(edited_midi_name)

# load the new midi file to verify changes
pm_humanized = pretty_midi.PrettyMIDI(edited_midi_name)

# convert the midi file to audio using a soundfont
wave = pm_humanized.fluidsynth(sf2_path="SalC5Light2.sf2", fs=sample_rate)

# save the audio as a wav file
sf.write("edited_midi.wav", wave, sample_rate)

# load the audio file into pydub for further processing
segment = AudioSegment.from_wav("edited_midi.wav")

# apply a low-pass filter to reduce high-frequency sharpness
filtered_segment = segment.low_pass_filter(2500)

# add subtle reverb effect by overlaying a quieter version of the same audio
filtered_segment = filtered_segment.overlay(filtered_segment - 6, position=50)

# normalize the audio to avoid overly loud or quiet sections
filtered_segment = filtered_segment.normalize()

# play the final processed audio
playback.play(filtered_segment)

print(f"edited midi saved as: {edited_midi_name}")