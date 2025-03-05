import os
from pypianoroll import read, to_pretty_midi
from pydub import AudioSegment, playback
import soundfile as sf
import numpy as np
import pretty_midi

# define sample rate for better audio quality
sample_rate = 44100

# find the most recent MIDI file in the directory
midi_files = [f for f in os.listdir() if f.endswith(".mid")]
if not midi_files:
    raise FileNotFoundError("No MIDI files found in the directory.")

# get the most recently created/modified MIDI file
latest_midi = max(midi_files, key=os.path.getctime)
original_file_name = os.path.splitext(latest_midi)[0]  # Remove .mid extension

print(f"Processing most recent MIDI file: {latest_midi}")

# load the MIDI file into a multitrack object
multitrack = read(latest_midi)

# convert the multitrack MIDI to PrettyMIDI format
pm = to_pretty_midi(multitrack)

# create a new PrettyMIDI object to store processed notes
new_pm = pretty_midi.PrettyMIDI()

# process each instrument in the MIDI file
for instrument in pm.instruments:
    new_instrument = pretty_midi.Instrument(program=instrument.program)
    
    # sort notes by start time
    instrument.notes.sort(key=lambda note: note.start)
    
    for note in instrument.notes:
        # preserve the original note timing and length
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end
        )
        new_instrument.notes.append(new_note)
    
    new_pm.instruments.append(new_instrument)

# create another PrettyMIDI object for additional effects
processed_pm = pretty_midi.PrettyMIDI()
for instrument in new_pm.instruments:
    processed_instrument = pretty_midi.Instrument(program=instrument.program)
    
    # sort notes by start time again
    instrument.notes.sort(key=lambda note: note.start)
    
    for note in instrument.notes:
        # introduce slight timing variation (-10ms to +10ms) for a more human feel
        time_variation = np.random.uniform(-0.01, 0.01)
        new_start = max(0, note.start + time_variation)
        new_end = max(new_start + 0.05, note.end + time_variation)

        # create the new processed note
        new_note = pretty_midi.Note(
            velocity=20,  # keep dynamics soft, 0 for softest and 127 for hardest
            pitch=note.pitch,
            start=new_start,
            end=new_end
        )
        processed_instrument.notes.append(new_note)
    
    processed_pm.instruments.append(processed_instrument)

# convert the midi file to audio using a soundfont
wave = processed_pm.fluidsynth(sf2_path=r"soundfonts/SalC5Light2.sf2", fs=sample_rate)

# save the audio as a wav file
sf.write("output.wav", wave, sample_rate)

# load the audio file into pydub for further processing
segment = AudioSegment.from_wav("output.wav")

# apply a low-pass filter to reduce high-frequency sharpness
filtered_segment = segment.low_pass_filter(2500)

# add subtle reverb effect by overlaying a quieter version of the same audio
filtered_segment = filtered_segment.overlay(filtered_segment - 6, position=50)

# normalize the audio to avoid overly loud or quiet sections
filtered_segment = filtered_segment.normalize()

# play the final processed audio
playback.play(filtered_segment)