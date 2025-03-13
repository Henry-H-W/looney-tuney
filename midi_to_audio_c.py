import os
from pypianoroll import read, to_pretty_midi
from pydub import AudioSegment, playback
import soundfile as sf
import numpy as np
import pretty_midi

def convert_midi_c(filename: str):
    sample_rate = 44100
    time_scaling_factor = 1.225  # Slow down factor for times after 15 sec
    split_time = 15.0  # Seconds; before this, no slow-down

    # Construct MIDI filename (assumes one file with the given name)
    midi_filename = filename + ".mid"
    if not os.path.exists(midi_filename):
        raise FileNotFoundError("MIDI file not found: " + midi_filename)

    print(f"Processing MIDI file: {midi_filename}")

    # Load the MIDI file into a multitrack object
    multitrack = read(midi_filename)

    # Convert the multitrack MIDI to PrettyMIDI format
    pm = to_pretty_midi(multitrack)

    # Get original tempo; if no tempo changes, default to 120 BPM.
    original_tempo = pm.estimate_tempo() if pm.get_tempo_changes()[1].size > 0 else 120.0
    # For times after split_time, we want an adjusted tempo.
    adjusted_tempo = original_tempo / time_scaling_factor  
    print(f"Original Tempo: {original_tempo}, Adjusted Tempo (after {split_time} sec): {adjusted_tempo}")

    # Create a new PrettyMIDI object.
    new_pm = pretty_midi.PrettyMIDI()

    # (Optional) Add tempo change events:
    # Set original tempo for the first 15 seconds and then change at split_time.
    original_tempo_event = pretty_midi.ControlChange(number=51, value=int(60000000 / original_tempo), time=0)
    adjusted_tempo_event = pretty_midi.ControlChange(number=51, value=int(60000000 / adjusted_tempo), time=split_time)

    for instrument in pm.instruments:
        new_instrument = pretty_midi.Instrument(program=0)

        # (For the first instrument, add the tempo events)
        if len(new_pm.instruments) == 0:
            new_instrument.control_changes.extend([original_tempo_event, adjusted_tempo_event])

        # Process each note with piecewise time mapping.
        for note in instrument.notes:
            # Case 1: Note entirely before split_time -> no scaling.
            if note.end <= split_time:
                new_start = note.start
                new_end = note.end
            # Case 2: Note entirely after split_time -> full slow-down.
            elif note.start >= split_time:
                new_start = split_time + (note.start - split_time) * time_scaling_factor
                new_end = split_time + (note.end - split_time) * time_scaling_factor
            # Case 3: Note spans the split_time.
            else:
                # Keep the part before split_time unslowed, and slow down the remainder.
                new_start = note.start
                new_end = split_time + (note.end - split_time) * time_scaling_factor

            # Optional: Add a slight random variation (-10ms to +10ms) for realism.
            time_variation = np.random.uniform(-0.01, 0.01)
            new_start = max(0, new_start + time_variation)
            new_end = max(new_start + 0.05, new_end + time_variation)

            # Create new note with adjusted timings.
            new_note = pretty_midi.Note(
                velocity=90,
                pitch=note.pitch,
                start=new_start,
                end=new_end
            )
            new_instrument.notes.append(new_note)

        new_pm.instruments.append(new_instrument)

    # Convert the modified MIDI to audio using a soundfont.
    soundfont_path = r"soundfonts/SalC5Light2.sf2"
    wave = new_pm.fluidsynth(sf2_path=soundfont_path, fs=sample_rate)

    # Save the unprocessed audio as a WAV file.
    sf.write(filename + "_unprocessed.wav", wave, sample_rate)

    # Load the audio file into pydub for further processing.
    segment = AudioSegment.from_wav(filename + "_unprocessed.wav")

    # Apply a low-pass filter to reduce high-frequency sharpness.
    filtered_segment = segment.low_pass_filter(500)

    # Add subtle reverb effect by overlaying a quieter version.
    filtered_segment = filtered_segment.overlay(filtered_segment - 6, position=50)

    # Normalize the audio.
    filtered_segment = filtered_segment.normalize()

    # Save the processed audio as a new WAV file.
    filtered_segment.export(filename + ".wav", format="wav")

def main():
    filename = input("Enter the MIDI filename (without extension): ").strip()
    try:
        convert_midi(filename)
        print("Conversion completed successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()