import os
from pypianoroll import read, to_pretty_midi
from pydub import AudioSegment, playback
import soundfile as sf
import numpy as np
import pretty_midi

def convert_midi():
    # Define sample rate for better audio quality
    sample_rate = 44100
    time_scaling_factor = 1.225  # Slow down factor

    # Find the most recent MIDI file in the directory
    midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the directory.")

    # Get the most recently created/modified MIDI file
    latest_midi = max(midi_files, key=os.path.getctime)
    original_file_name = os.path.splitext(latest_midi)[0]

    print(f"Processing most recent MIDI file: {latest_midi}")

    # Load the MIDI file into a multitrack object
    multitrack = read(latest_midi)

    # Convert the multitrack MIDI to PrettyMIDI format
    pm = to_pretty_midi(multitrack)

    # Get original tempo, ensuring there's a fallback value
    original_tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
    adjusted_tempo = original_tempo / time_scaling_factor  # Adjust tempo for slow-down
    print(f"Original Tempo: {original_tempo}, Adjusted Tempo: {adjusted_tempo}")

    # Create a new PrettyMIDI object
    new_pm = pretty_midi.PrettyMIDI()

    # **Add a Tempo Change Event at Time 0**
    tempo_change = pretty_midi.ControlChange(number=51, value=int(60000000 / adjusted_tempo), time=0)

    # Process each instrument in the MIDI file
    for instrument in pm.instruments:
        new_instrument = pretty_midi.Instrument(program=0)

        # Add tempo control change to the first instrument
        if len(new_pm.instruments) == 0:
            new_instrument.control_changes.append(tempo_change)

        # Sort notes by start time
        instrument.notes.sort(key=lambda note: note.start)

        for note in instrument.notes:
            # Scale both start time and duration
            new_start = note.start * time_scaling_factor
            new_duration = max(0.05, (note.end - note.start) * time_scaling_factor)
            new_end = new_start + new_duration

            # Slight timing variation (-10ms to +10ms) for realism
            time_variation = np.random.uniform(-0.01, 0.01)
            new_start = max(0, new_start + time_variation)
            new_end = max(new_start + 0.05, new_end + time_variation)

            # Create new note
            new_note = pretty_midi.Note(
                velocity=90,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)

        new_pm.instruments.append(new_instrument)

    # Convert the modified MIDI to audio using a soundfont
    soundfont_path = r"soundfonts/SalC5Light2.sf2"
    wave = new_pm.fluidsynth(sf2_path=soundfont_path, fs=sample_rate)

    # Save the audio as a WAV file
    sf.write("output.wav", wave, sample_rate)

    # Load the audio file into pydub for further processing
    segment = AudioSegment.from_wav("output.wav")

    # Apply a low-pass filter to reduce high-frequency sharpness
    filtered_segment = segment.low_pass_filter(500)

    # Add subtle reverb effect by overlaying a quieter version of the same audio
    filtered_segment = filtered_segment.overlay(filtered_segment - 6, position=50)

    # Normalize the audio to avoid overly loud or quiet sections
    filtered_segment = filtered_segment.normalize()

    # Play the final processed audio
    # playback.play(filtered_segment)