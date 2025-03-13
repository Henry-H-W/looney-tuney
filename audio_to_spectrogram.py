import os
import sys
import time
import random
import threading
import cmasher as cmr  # For additional colormaps

import numpy as np
import pygame
from pygame import mixer
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from scipy.ndimage import gaussian_filter

from generate import generate
from generateTTE import generate_music
from midi_to_audio import convert_midi
from delete_files import delete_generated_files
from collaborateTTE import extend_midi
from collaborate import generate_collab

from explosion_animation import Firework
from mido import Message, MidiFile, MidiTrack
import rtmidi

# -------------------- Global MIDI and Recording Variables --------------------
# These globals let us re-use the same MIDI instance.
midi_in = None
selected_port_index = None
# At the top, define an additional global flag:
recording_finished = False
processing_started = False

# Variables for recording incoming MIDI messages
is_recording = False
recorded_midi_messages = []  # Each entry is (message_bytes, relative_time)
recording_start_time = None

# -------------------- Helper Functions --------------------
def list_midi_ports():
    """Return a list of available MIDI input port names."""
    midi_temp = rtmidi.MidiIn()
    return midi_temp.get_ports()

def initialize_midi():
    """Initialize (or re-use) the global MIDI input instance."""
    global midi_in, selected_port_index
    if midi_in is None:
        midi_in = rtmidi.MidiIn()
        ports = midi_in.get_ports()
        if ports:
            selected_port_index = 0  # Default to the first available MIDI port
            midi_in.open_port(selected_port_index)
            print(f"Connected to MIDI device: {ports[selected_port_index]}")
        else:
            print("No MIDI devices found.")
            selected_port_index = None

def start_recording(duration=15):
    """
    Record incoming MIDI messages for `duration` seconds using the global
    MIDI input instance. After recording, write the messages to a MIDI file
    and process them.
    """
    global recording_finished, is_recording, recorded_midi_messages, recording_start_time
    recorded_midi_messages = []
    recording_start_time = None
    is_recording = True
    print(f"Recording MIDI for {duration} seconds...")
    time.sleep(duration)
    is_recording = False
    print("Recording finished. Saving to MIDI file...")

    # Build a MIDI file from recorded messages
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    last_time = 0
    for msg_bytes, t in recorded_midi_messages:
        delta_time = int((t - last_time) * 1000)  # delta time in milliseconds
        mido_msg = Message.from_bytes(msg_bytes, time=delta_time)
        track.append(mido_msg)
        last_time = t

    midi_file.save("recording.mid")
    print("MIDI recording saved as recording.mid")
    # Find the most recent MIDI file in the directory
    midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    if not midi_files:
        raise FileNotFoundError("No MIDI files found in the directory.")

    # Get the most recently created/modified MIDI file
    latest_midi = max(midi_files, key=os.path.getctime)
    print(f"Extending MIDI file: {latest_midi}")

    extend_midi('recording.mid', 'collab_output.mid', additional_intervals=15) # algorithmic extension
    # generate_collab() # ai extension

    # Once collab output is created, signal that fireworks should stop updating.
    recording_finished = True

    process_audio_and_start('collab_output')


# ----- Utility: Convert a Matplotlib colormap to a lookup table for pygame -----
def generatePgColormap(cm_name):
    """
    Converts a matplotlib colormap to a lookup table (LUT) of shape (256, 4).
    Only the RGB channels will be used.
    """
    colormap = plt.get_cmap(cm_name)
    colormap._init()  # initialize the LUT
    lut = (colormap._lut * 255).view(np.ndarray).astype(np.uint8)
    return lut

# -------------------- Constants and Parameters --------------------
CHUNKSIZE = 2048        # Base chunk size (samples)
N_FFT = 4096            # FFT length for high resolution
WATERFALL_DURATION = 10  # seconds to show in waterfall
OVERLAP_RATIO = 0.5     # 50% overlap for smoother scrolling
chunk_size = int(CHUNKSIZE * (1 - OVERLAP_RATIO))  # step size per update
EPS = 1e-8
spectrogram_active = False

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

# Frequency vector (only keep frequencies up to FREQ_LIMIT)
FREQ_LIMIT = 4000  # Hz (adjust as needed)

# -------------------- Pygame Setup --------------------
pygame.init()
window_width, window_height = pygame.display.Info().current_w, pygame.display.Info().current_h
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Live Waterfall Spectrogram (Audio File)")
boldfont = pygame.font.SysFont("latoblack", 34)
font = pygame.font.SysFont("latolight", 34)
buttonfont = pygame.font.SysFont("latolight", 25)
clock = pygame.time.Clock()

# Load PNG overlays
overlay_image1 = pygame.image.load(r"gui_assets\logo.png").convert_alpha()
overlay_image1 = pygame.transform.scale(overlay_image1, (150, 150))
overlay_image2 = pygame.image.load(r"gui_assets\slur.png").convert_alpha()
overlay_image2 = pygame.transform.scale(overlay_image2, (30, 8.5))

# Define button properties
button_width, button_height = 200, 50
button1_1 = pygame.Rect(50, 270, button_width, button_height)
button1_2 = pygame.Rect(250, 270, button_width, button_height)
button_generate = pygame.Rect(50, 720, button_width, button_height)
button_dropdown = pygame.Rect(50, 400, button_width * 2.5, button_height)
button_record = pygame.Rect(50, 650, button_width, button_height)

# Track which button is active
active_button1 = None  # "generation" or "collaboration"
active_button2 = None  # "generate", "dropdown", or "record"
text3_color = WHITE  # Default color
text5_color = WHITE

# MIDI port dropdown options
midi_ports = list_midi_ports()
if not midi_ports:
    midi_ports = ["No MIDI devices found"]
dropdown_options = midi_ports
option = "No upload port provided"

# -------------------- Colormap Setup --------------------
COLORMAPS = [
    "inferno", "magma",  # Standard colormaps
    "gist_heat", "gray", "copper", "bone", "cubehelix",  # Extra colormaps
    # cmasher colormaps
    "cmr.lavender", "cmr.tree", "cmr.dusk", "cmr.nuclear", "cmr.emerald",
    "cmr.sapphire", "cmr.cosmic", "cmr.ember", "cmr.toxic",
    "cmr.lilac", "cmr.sepia", "cmr.amber", "cmr.eclipse",
    "cmr.ghostlight", "cmr.arctic", "cmr.jungle",
    "cmr.swamp", "cmr.freeze", "cmr.amethyst", "cmr.flamingo",
    "cmr.savanna", "cmr.sunburst", "cmr.voltage", "cmr.gothic",
    "cmr.apple", "cmr.torch", "cmr.rainforest", "cmr.chroma"
]
selected_colormap = random.choice(COLORMAPS)
print(f"Selected Colormap: {selected_colormap}")
if selected_colormap.startswith("cmr."):
    selected_colormap = getattr(cmr, selected_colormap.split(".")[1])
    lut = generatePgColormap(selected_colormap)
else:
    lut = generatePgColormap(selected_colormap)
rgb_lut = lut[:, :3]  # Use only the RGB channels

# -------------------- Audio Playback Globals --------------------
start_time = None  # Will be set when playback starts

def play_audio(filename: str):
    global start_time, audio_finished
    temp_file = filename + ".wav"

    # Export the new audio
    audio.export(temp_file, format="wav")

    mixer.init()
    mixer.music.load(temp_file)
    mixer.music.play()
    start_time = time.time()

    while mixer.music.get_busy():
        time.sleep(0.1)

    mixer.music.stop()
    mixer.quit()
    audio_finished = True

# -------------------- Dropdown and Button Fade Setup --------------------
dropdown_options = ["DeepMind 12"]
dropdown_height = 50
dropdown_active = False
dropdown_rects = []

button_generate_alpha = 0
button_dropdown_alpha = 0
button_record_alpha = 0
fade_in_speed = 50
button_generate_visible = False
button_dropdown_visible = False

button_generate_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
button_record_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
button_dropdown_surface = pygame.Surface((button_width * 2.5, button_height), pygame.SRCALPHA)

def render_fading_text(text, font, color, alpha):
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)
    return text_surface

def draw_dropdown_menu():
    global dropdown_rects
    pygame.draw.rect(screen, WHITE, pygame.Rect(button_dropdown.x, button_dropdown.y + button_height, button_dropdown.width, dropdown_height * len(dropdown_options)))
    dropdown_rects = []
    for i, opt in enumerate(dropdown_options):
        rect = pygame.Rect(button_dropdown.x, button_dropdown.y + button_height + i * dropdown_height, button_dropdown.width, dropdown_height)
        dropdown_rects.append(rect)
        pygame.draw.rect(screen, BLACK, rect)
        pygame.draw.rect(screen, WHITE, rect, 1)
        text = buttonfont.render(opt, True, WHITE)
        screen.blit(text, (65, rect.y + (rect.height - text.get_height()) // 2))

def fade_in(button, button_visible, button_surface, button_alpha, active_button, activity, button_width, button_height, text):
    button_surface.fill((0, 0, 0, 0))
    if active_button == activity:
        pygame.draw.rect(button_surface, (255, 255, 255, button_alpha), (0, 0, button_width, button_height))
        pygame.draw.rect(button_surface, (0, 0, 0, button_alpha), (0, 0, button_width, button_height), 2)
        colour = BLACK
    else:
        pygame.draw.rect(button_surface, (0, 0, 0, button_alpha), (0, 0, button_width, button_height))
        pygame.draw.rect(button_surface, (255, 255, 255, button_alpha), (0, 0, button_width, button_height), 2)
        colour = WHITE
    text1_surface = render_fading_text(text, buttonfont, colour, button_alpha)
    button_surface.blit(text1_surface, ((button_width - text1_surface.get_width()) // 2,
                                          (button_height - text1_surface.get_height()) // 2))
    screen.blit(button_surface, (button.x, button.y))
    if button_visible:
        if button_alpha < 255:
            button_alpha += fade_in_speed
            button_alpha = min(button_alpha, 255)
    else:
        if button_alpha > 0:
            button_alpha -= fade_in_speed
            button_alpha = max(button_alpha, 0)
    return button_alpha

def process_audio_and_start(filename: str):
    convert_midi(filename)
    audio_file = filename + ".wav"
    file_ext = os.path.splitext(audio_file)[1][1:]
    global audio, sample_rate, samples, WATERFALL_FRAMES, FREQ_VECTOR, freq_mask, num_freq_bins, waterfall_image_data
    audio = AudioSegment.from_file(audio_file, format=file_ext).set_channels(1)
    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if np.max(np.abs(samples)) > 0:
        samples /= np.max(np.abs(samples))
    WATERFALL_FRAMES = int(WATERFALL_DURATION * sample_rate / chunk_size)
    FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=1/sample_rate)
    freq_mask = FREQ_VECTOR <= FREQ_LIMIT
    FREQ_VECTOR = FREQ_VECTOR[freq_mask]
    num_freq_bins = len(FREQ_VECTOR)
    waterfall_image_data = np.full((num_freq_bins, WATERFALL_FRAMES), -10, dtype=np.float32)
    global current_index, start_time, audio_finished, spectrogram_active
    current_index = 0
    audio_finished = False
    audio_thread = threading.Thread(target=play_audio, args=(filename,), daemon=True)
    audio_thread.start()
    spectrogram_active = True

# -------------------- Fireworks and Parent Object --------------------
fireworks = []
hue_index = 0.0

class Parent:
    pass

parent_obj = Parent()
parent_obj.window_size = (window_width, window_height)

# -------------------- Main Loop --------------------
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button1_1.collidepoint(event.pos):
                print("Generation button clicked!")
                active_button1 = "generation"
                active_button2 = None
                text3_color = WHITE
                dropdown_active = False
                button_generate_visible = True
                button_dropdown_visible = False
            elif button_generate.collidepoint(event.pos) and active_button1 == "generation":
                active_button2 = "generate"
                print("Generate button clicked")
                def generate_midi():
                    # generate('generated_output.mid')
                    random_scale = random.randint(59, 70)  # random root note within a musical range
                    generate_music(random_scale, 'generated_output.mid') # provide the option of algorithmic generation
                    process_audio_and_start('generated_output')
                threading.Thread(target=generate_midi, daemon=True).start()
            elif button1_2.collidepoint(event.pos):
                print("Collaboration button clicked!")
                active_button1 = "collaboration"
                active_button2 = None
                text5_color = WHITE
                button_dropdown_visible = True
                button_generate_visible = False
                initialize_midi()
            elif button_dropdown.collidepoint(event.pos) and active_button1 == "collaboration":
                active_button2 = "dropdown"
                dropdown_active = not dropdown_active
                print("Dropdown button clicked!")
            elif button_record.collidepoint(event.pos) and active_button1 == "collaboration":
                active_button2 = "record"
                dropdown_active = False
                print("Record button clicked!")
            if dropdown_active:
                for i, rect in enumerate(dropdown_rects):
                    if rect.collidepoint(event.pos):
                        option = dropdown_options[i]
                        selected_port_index = i
                        print(f"Dropdown Option {i} selected: {option}")
                        dropdown_active = False
                        active_button2 = None
            elif button_record.collidepoint(event.pos) and active_button1 == "collaboration":
                if selected_port_index is not None and dropdown_options[selected_port_index] != "No MIDI devices found":
                    print("Starting MIDI recording...")
                    threading.Thread(target=start_recording, args=(15,), daemon=True).start()
                else:
                    print("No valid MIDI port selected.")

    screen.fill(BLACK)

    # Update waterfall spectrogram if audio playback has started
    if start_time is not None and spectrogram_active:
        elapsed_time = time.time() - start_time
        if not audio_finished:
            samples_played = int(elapsed_time * audio.frame_rate)
            current_index = min(samples_played, len(samples) - chunk_size)
            if current_index + chunk_size < len(samples):
                data = samples[current_index: current_index + chunk_size]
                current_index += chunk_size
            else:
                data = np.zeros(chunk_size, dtype=np.float32)
        else:
            data = np.zeros(chunk_size, dtype=np.float32)
            recording_finished = False
            active_button2 = None

        windowed = np.hanning(len(data)) * data
        X = np.abs(np.fft.rfft(windowed, n=N_FFT))[freq_mask]
        new_frame = np.log10(X + 1e-12)
        waterfall_image_data = np.roll(waterfall_image_data, -1, axis=1)
        waterfall_image_data[:, -1] = new_frame
        waterfall_image_data[waterfall_image_data < -10] = -10
        waterfall_blurred = gaussian_filter(waterfall_image_data, sigma=0.75)
        min_val = waterfall_blurred.min()
        max_val = waterfall_blurred.max()
        if max_val - min_val > 0:
            normalized = (waterfall_blurred - min_val) / (max_val - min_val)
        else:
            normalized = waterfall_blurred - min_val
        normalized = (normalized * 255).astype(np.uint8)
        brightness_factor = 0.5
        color_image = (rgb_lut[normalized] * brightness_factor).astype(np.uint8)
        surf_array = np.transpose(color_image, (1, 0, 2))
        spec_surface = pygame.surfarray.make_surface(surf_array)
        spec_surface = pygame.transform.scale(spec_surface, (window_width, window_height))
        screen.blit(spec_surface, (0, 0))

    # Overlay images and text
    screen.blit(overlay_image1, (0, 10))
    text1 = boldfont.render("Welcome to MAiSTRO: The Classical Music Composition Model!", True, WHITE)
    screen.blit(text1, (50, 150))
    text2 = font.render("Start by choosing between generation mode or collaboration mode:", True, WHITE)
    screen.blit(text2, (50, 200))

    # Draw buttons with dynamic colors
    if active_button1 == "generation":
        pygame.draw.rect(screen, WHITE, button1_1)
        pygame.draw.rect(screen, WHITE, button1_1, 2)
        text1_color = BLACK
        button_generate_alpha = fade_in(button_generate, button_generate_visible, button_generate_surface, button_generate_alpha, active_button2, "generate", button_width, button_height, "Generate")
        button_dropdown_alpha = fade_in(button_dropdown, button_dropdown_visible, button_dropdown_surface, button_dropdown_alpha, active_button2, "dropdown", button_width * 2.5, button_height, option)
        button_record_alpha = fade_in(button_record, button_dropdown_visible, button_record_surface, button_record_alpha, active_button2, "record", button_width, button_height, "Record")
    else:
        pygame.draw.rect(screen, BLACK, button1_1)
        pygame.draw.rect(screen, WHITE, button1_1, 2)
        text1_color = WHITE

    if active_button1 == "collaboration":
        pygame.draw.rect(screen, WHITE, button1_2)
        pygame.draw.rect(screen, WHITE, button1_2, 2)
        text2_color = BLACK

        if selected_port_index is not None and dropdown_options[selected_port_index] != "No MIDI devices found":
            # Poll the global MIDI input for messages
            if midi_in and selected_port_index is not None:
                msg = midi_in.get_message()
                if msg:
                    message, delta_time = msg
                    if len(message) >= 3:
                        note = message[1]
                        velocity = message[2]
                        if message[0] == 149 and velocity > 0:
                            x_pos = (note - 36) / float(96 - 36)
                            x_pos = max(0.1, min(0.9, x_pos))
                            grey_value = random.randint(50, 255)
                            color = (grey_value, grey_value, grey_value)
                            fireworks.append(Firework(parent_obj, x_pos, color, intensity=1))
                            hue_index += 0.08
                            if hue_index > 1.0:
                                hue_index = 0.0
                            
                        # Also record the MIDI message if recording is active
                        if is_recording:
                            current_time = time.time()
                            if recording_start_time is None:
                                recording_start_time = current_time
                            relative_time = current_time - recording_start_time
                            recorded_midi_messages.append((message, relative_time))

            if recording_finished:
                fireworks.clear()
            else:
                for fw in fireworks:
                    fw.Update()
                for fw in fireworks:
                    fw.Draw(screen)
            pygame.draw.rect(screen, WHITE, button_record)
            pygame.draw.rect(screen, BLACK, button_record, 2)
            text_record = buttonfont.render("Record", True, BLACK)
            screen.blit(text_record, (
                button_record.x + (button_record.width - text_record.get_width()) // 2,
                button_record.y + (button_record.height - text_record.get_height()) // 2
            ))
        button_dropdown_alpha = fade_in(button_dropdown, button_dropdown_visible, button_dropdown_surface, button_dropdown_alpha, active_button2, "dropdown", button_width * 2.5, button_height, option)
        button_generate_alpha = fade_in(button_generate, button_generate_visible, button_generate_surface, button_generate_alpha, active_button2, "generate", button_width, button_height, "Generate")
        button_record_alpha = fade_in(button_record, button_dropdown_visible, button_record_surface, button_record_alpha, active_button2, "record", button_width, button_height, "Record")
    else:
        pygame.draw.rect(screen, BLACK, button1_2)
        pygame.draw.rect(screen, WHITE, button1_2, 2)
        text2_color = WHITE

    text1 = buttonfont.render("Generation", True, text1_color)
    text2 = buttonfont.render("Collaboration", True, text2_color)
    text4_surface = render_fading_text("Connect your MIDI device", font, WHITE, button_dropdown_alpha)
    screen.blit(text4_surface, (50, 350))
    screen.blit(text1, (button1_1.x + (button_width - text1.get_width()) // 2, button1_1.y + (button_height - text1.get_height()) // 2))
    screen.blit(text2, (button1_2.x + (button_width - text2.get_width()) // 2, button1_2.y + (button_height - text2.get_height()) // 2))

    if dropdown_active:
        draw_dropdown_menu()

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
