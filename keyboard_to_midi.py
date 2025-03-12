import mido
from mido import MidiFile, MidiTrack, Message
import rtmidi
import time

# IMPORTANT: as soon as you confirm your MIDI port, start playing! for some
# reason, if you start a bit later the midi will still record, but the 
# processing later gets messed up (in midi_to_audio.py) unless you play 
# as soon as you're allowed to... weird i know

# you might have to reinstall python-rtmidi if it doesn't work

# Function to list available MIDI input ports
def list_midi_ports():
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    if not ports:
        print("No MIDI input devices found.")
        return None
    for i, port in enumerate(ports):
        print(f"{i}: {port}")
    return ports

# Function to record MIDI input
def record_midi(input_port, output_filename="recorded.mid", duration=20):
    print(f"Recording MIDI for {duration} seconds...")
    
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(input_port)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        msg_and_dt = midi_in.get_message()
        if msg_and_dt:
            msg, delta_time = msg_and_dt
            track.append(Message.from_bytes(msg, time=int(delta_time * 1000)))
        else:
            time.sleep(0.01)
    
    midi_in.close_port()
    
    # Save MIDI file
    midi_file.save(output_filename)
    print(f"MIDI recording saved as {output_filename}")

# Main script execution
if __name__ == "__main__":
    ports = list_midi_ports()
    if ports:
        try:
            port_number = int(input("Select MIDI input port number: "))
            if port_number < 0 or port_number >= len(ports):
                print("Invalid port number.")
            else:
                record_midi(port_number, "recording.mid", duration=20)
        except ValueError:
            print("Please enter a valid number.")