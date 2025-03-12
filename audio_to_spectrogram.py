import os
import sys
import time
import random
import threading
import cmasher as cmr  # Import cmasher for additional colormaps

import numpy as np
import pygame
from pygame import mixer
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from scipy.ndimage import gaussian_filter

# from generateTTE import generate_music
from generate import generate
from midi_to_audio import convert_midi
from delete_files import delete_generated_files
import rtmidi
from mido import Message, MidiFile, MidiTrack
from collaborateTTE import extend_midi
from collaborate import generate_collab
# ------- Helper functions for visual aids for keyboard input ----------
def list_midi_ports():
    """Return a list of available MIDI input port names."""
    midi_in = rtmidi.MidiIn()
    return midi_in.get_ports()

def record_midi(port_number, output_filename="recording.mid", duration=15):
    """
    Records MIDI from the chosen port_number for `duration` seconds,
    and saves it to output_filename.
    """
    print(f"Recording MIDI for {duration} seconds on port {port_number}...")

    # Set up rtmidi
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(port_number)

    # Create a new MidiFile with a single track
    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    start_time = time.time()
    last_time = start_time

    while time.time() - start_time < duration:
        msg_and_dt = midi_in.get_message()
        if msg_and_dt:
            msg, delta_time = msg_and_dt
            # Convert the rtmidi message to Mido Message
            # The first 3 bytes are usually [status, note, velocity]
            # rtmidi returns delta_time in seconds as well
            mido_msg = Message.from_bytes(msg, time=int(delta_time * 1000))
            track.append(mido_msg)
        else:
            time.sleep(0.01)

    midi_in.close_port()
    midi_file.save(output_filename)
    print(f"MIDI recording saved as {output_filename}")

def do_record():
    global is_recording
    is_recording = True
    record_midi(selected_port_index, "recording.mid", duration=15)
    is_recording = False
    print("Recording finished. Check recording.mid!")
    # Find the most recent MIDI file in the directory
    # midi_files = [f for f in os.listdir() if f.endswith(".mid")]
    # if not midi_files:
    #     raise FileNotFoundError("No MIDI files found in the directory.")

    # Get the most recently created/modified MIDI file
    # latest_midi = max(midi_files, key=os.path.getctime)
    # print(f"Extending MIDI file: {latest_midi}")

    # extend_midi('recording.mid', 'collab_output.mid', additional_intervals=30)
    generate_collab()

# ----- Utility: Convert a Matplotlib colormap to a lookup table for pygame -----
def generatePgColormap(cm_name):
    """
    Converts a matplotlib colormap to a lookup table (LUT) of shape (256, 4).
    We will later use only the RGB channels.
    """
    colormap = plt.get_cmap(cm_name)
    colormap._init()  # initialize the LUT
    lut = (colormap._lut * 255).view(np.ndarray).astype(np.uint8)
    return lut

# ----- Constants and Parameters -----
CHUNKSIZE = 2048        # Base chunk size (samples)
N_FFT = 4096           # FFT length for high resolution
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
FREQ_LIMIT = 4000  # Hz (change as needed)

# ----- Setup Pygame -----
pygame.init()
window_width, window_height = pygame.display.Info().current_w, pygame.display.Info().current_h # - 75
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Live Waterfall Spectrogram (Audio File)")
boldfont = pygame.font.SysFont("latoblack", 34)
font = pygame.font.SysFont("latolight", 34)
buttonfont = pygame.font.SysFont("latolight", 25)
clock = pygame.time.Clock()

# Load the PNG overlay
overlay_image1 = pygame.image.load(r"gui_assets\logo.png").convert_alpha()  # Preserve transparency
overlay_image1 = pygame.transform.scale(overlay_image1, (150, 150))  # Resize if needed
overlay_image2 = pygame.image.load(r"gui_assets\slur.png").convert_alpha()  # Preserve transparency
overlay_image2 = pygame.transform.scale(overlay_image2, (30, 8.5))  # Resize if needed (50, 17)

# Define button properties
button_width, button_height = 200, 50
button1_1 = pygame.Rect(50, 270, button_width, button_height)
button1_2 = pygame.Rect(250, 270, button_width, button_height)
button_generate = pygame.Rect(50, 720, button_width, button_height)
button_dropdown = pygame.Rect(50, 400, button_width*2.5, button_height)
button_record = pygame.Rect(50, 650, button_width, button_height)

# Track which button is active
active_button1 = None  # Can be "generation" or "collaboration"
active_button2 = None # Can be "generate" or "dropdown" or "record"

text3_color = WHITE  # Default color
text5_color = WHITE
is_recording = False

midi_ports = list_midi_ports()
if not midi_ports:
    midi_ports = ["No MIDI devices found"]
dropdown_options = midi_ports  # use actual port list in the dropdown
selected_port_index = None
option = "No upload port provided"

# ----- Select a random colormap and generate its LUT -----
COLORMAPS = [
    "inferno", "magma",  # Standard colormaps
    "gist_heat", "gray", "copper", "bone", "cubehelix",  # Extra colormaps
    # cmasher colormaps ------------------------------------------------
    "cmr.lavender", "cmr.tree", "cmr.dusk", "cmr.nuclear", "cmr.emerald",
    "cmr.sapphire", "cmr.cosmic", "cmr.ember", "cmr.toxic",
    "cmr.lilac", "cmr.sepia", "cmr.amber", "cmr.eclipse",
    "cmr.ghostlight", "cmr.arctic", "cmr.jungle",
    "cmr.swamp", "cmr.freeze", "cmr.amethyst", "cmr.flamingo",
    "cmr.savanna", "cmr.sunburst", "cmr.voltage", "cmr.gothic",
    "cmr.apple", "cmr.torch", "cmr.rainforest", "cmr.chroma"
]

# Select a random colormap
selected_colormap = random.choice(COLORMAPS)
print(f"Selected Colormap: {selected_colormap}")

# Use cmasher for custom colormaps
if selected_colormap.startswith("cmr."):
    selected_colormap = getattr(cmr, selected_colormap.split(".")[1])  # Get the colormap function
    lut = generatePgColormap(selected_colormap)
else:
    lut = generatePgColormap(selected_colormap)
rgb_lut = lut[:, :3]  # use only R,G,B channels

# ----- Audio Playback Globals -----
start_time = None  # will be set when playback starts

def play_audio(filename: str):
    global start_time, audio_finished
    temp_file = filename + ".wav"

    # Export the new audio
    audio.export(temp_file, format="wav")

    # Initialize pygame mixer and load the new file
    mixer.init()
    mixer.music.load(temp_file)

    # Play the new file
    mixer.music.play()

    # Set the playback start time
    start_time = time.time()

    # Wait until playback finishes
    while mixer.music.get_busy():
        time.sleep(0.1)

    # Stop and unload the mixer to release the file
    mixer.music.stop()
    mixer.quit()  # Ensures the file is no longer locked
    audio_finished = True

    # Delete temp files after playback
    # delete_generated_files()

# Define dropdown menu options
dropdown_options = ["DeepMind 12"]
dropdown_height = 50  # height of each option
dropdown_active = False  # Whether dropdown is active or not
dropdown_rects = []  # Keep track of the option rectangles
option = "No upload port provided"

# Fade-in button alpha values
button_generate_alpha = 0
button_dropdown_alpha = 0
button_record_alpha = 0
fade_in_speed = 50
button_generate_visible = False
button_dropdown_visible = False

# Create transparent surfaces for fading buttons
button_generate_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
button_record_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
button_dropdown_surface = pygame.Surface((button_width*2.5, button_height), pygame.SRCALPHA)

# Function to render text with transparency
def render_fading_text(text, font, color, alpha):
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)  # Apply transparency
    return text_surface

def draw_dropdown_menu():
    global dropdown_rects
    # Draw a background for the dropdown menu
    pygame.draw.rect(screen, WHITE, pygame.Rect(button_dropdown.x, button_dropdown.y + button_height, button_dropdown.width, dropdown_height * len(dropdown_options)))
    
    dropdown_rects = []  # Clear previous option rectangles
    
    # Draw each option in the dropdown menu
    for i, option in enumerate(dropdown_options):
        rect = pygame.Rect(button_dropdown.x, button_dropdown.y + button_height + i * dropdown_height, button_dropdown.width, dropdown_height)
        dropdown_rects.append(rect)
        
        # Draw the option box with a black outline
        pygame.draw.rect(screen, BLACK, rect)  # Draw black border
        pygame.draw.rect(screen, WHITE, rect, 1)  # Add a white border (2-pixel width)
        
        # Render the text and center it inside the option box
        text = buttonfont.render(option, True, WHITE)
        screen.blit(text, (65, rect.y + (rect.height - text.get_height()) // 2))

def fade_in(button, button_visible, button_surface, button_alpha, active_button, activity, button_width, button_height, text):
    # Clear the surface and apply alpha
    button_surface.fill((0, 0, 0, 0))  # Fully transparent background

    if active_button == activity:
        pygame.draw.rect(button_surface, (255, 255, 255, button_alpha), (0, 0, button_width, button_height))
        pygame.draw.rect(button_surface, (0, 0, 0, button_alpha), (0, 0, button_width, button_height), 2)
        colour = BLACK
    else:
        pygame.draw.rect(button_surface, (0, 0, 0, button_alpha), (0, 0, button_width, button_height))
        pygame.draw.rect(button_surface, (255, 255, 255, button_alpha), (0, 0, button_width, button_height), 2)
        colour = WHITE

    # Render text with fading effect
    text1_surface = render_fading_text(text, buttonfont, colour, button_alpha)

    # Center text inside the button surface
    button_surface.blit(text1_surface, ((button_width - text1_surface.get_width()) // 2,
                                        (button_height - text1_surface.get_height()) // 2))

    # Blit the transparent button onto the main screen
    screen.blit(button_surface, (button.x, button.y))

    # Smoothly fade in or out
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
    # Generate and convert MIDI (heavy processing)
    # generate_music(64, 'generated_output.mid') OLD GENERATION METHOD
    # generate('generated_output.mid')
    convert_midi(filename,)
    
    # Reload the new audio file (assumes it’s now the most recent file)
    # audio_files = [f for f in os.listdir() if f.endswith((".wav", ".mp3"))]
    audio_file = filename + ".wav"
    file_ext = os.path.splitext(audio_file)[1][1:]
    global audio, sample_rate, samples, WATERFALL_FRAMES, FREQ_VECTOR, freq_mask, num_freq_bins, waterfall_image_data
    audio = AudioSegment.from_file(audio_file, format=file_ext).set_channels(1)
    sample_rate = audio.frame_rate

    # Convert audio samples to a normalized numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if np.max(np.abs(samples)) > 0:
        samples /= np.max(np.abs(samples))
    
    # Compute waterfall dimensions with the new sample_rate:
    WATERFALL_FRAMES = int(WATERFALL_DURATION * sample_rate / chunk_size)
    FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=1/sample_rate)
    freq_mask = FREQ_VECTOR <= FREQ_LIMIT
    FREQ_VECTOR = FREQ_VECTOR[freq_mask]
    num_freq_bins = len(FREQ_VECTOR)
    
    # Reinitialize the waterfall image data
    waterfall_image_data = np.full((num_freq_bins, WATERFALL_FRAMES), -10, dtype=np.float32)
    
    # Reset playback globals
    global current_index, start_time, audio_finished, spectrogram_active
    current_index = 0
    # start_time = time.time()
    audio_finished = False
    
    # Start audio playback in a new thread
    audio_thread = threading.Thread(target=play_audio, args=(filename,), daemon=True)
    audio_thread.start()
    
    # Enable the spectrogram updates
    spectrogram_active = True

# ----- Main Loop -----
running = True
while running:
    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:  # Check if a key was pressed
            if event.key == pygame.K_ESCAPE:  # If ESCAPE key is pressed
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
                print("Generate button Clicked")
                # Start processing in a separate thread
                def generate_midi():
                    generate('generated_output.mid')
                    process_audio_and_start('generated_output',)
                processing_thread = threading.Thread(target=generate_midi, daemon=True)
                processing_thread.start()
            elif button1_2.collidepoint(event.pos):
                print("Collaboration button clicked!")
                active_button1 = "collaboration"
                active_button2 = None
                text5_color = WHITE
                button_dropdown_visible = True
                button_generate_visible = False
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
                        # The user picked the i-th option from the dropdown
                        option = dropdown_options[i]
                        selected_port_index = i
                        print(f"Dropdown Option {i} selected: {option}")
                        dropdown_active = False
                        active_button2 = None
            elif button_record.collidepoint(event.pos) and active_button1 == "collaboration":
                if selected_port_index is not None and dropdown_options[selected_port_index] != "No MIDI devices found":
                    print("Starting MIDI recording...")
                    def recording_and_processing():
                        do_record()  # Wait for recording to complete
                        process_audio_and_start("collab_output",)  # Start audio processing AFTER recording finishes

                    # Start the combined function in a separate thread
                    recording_thread = threading.Thread(target=recording_and_processing, daemon=True)
                    recording_thread.start()
                else:
                    print("No valid MIDI port selected.")

    screen.fill(BLACK)

    # Only update waterfall if playback has started
    if start_time is not None and spectrogram_active:
        # Compute elapsed time (only used while audio is active)
        elapsed_time = time.time() - start_time

        if not audio_finished:
            # While audio is playing, update current_index based on elapsed time
            samples_played = int(elapsed_time * audio.frame_rate)  # Actual samples played
            current_index = min(samples_played, len(samples) - chunk_size)  # Ensure we don’t go out of bounds

            if current_index + chunk_size < len(samples):
                data = samples[current_index: current_index + chunk_size]
                current_index += chunk_size
            else:
                data = np.zeros(chunk_size, dtype=np.float32)
        else:
            # Once audio has finished, use silence
            data = np.zeros(chunk_size, dtype=np.float32)
            active_button2 = None

        # Compute FFT using a Hanning window; apply the frequency mask
        windowed = np.hanning(len(data)) * data
        X = np.abs(np.fft.rfft(windowed, n=N_FFT))[freq_mask]
        new_frame = np.log10(X + 1e-12)  # take log (avoid log(0))
        
        # Scroll the waterfall image left by one column and add the new frame at the right
        waterfall_image_data = np.roll(waterfall_image_data, -1, axis=1)
        waterfall_image_data[:, -1] = new_frame

        # Force very low values to -10 (pure black background)
        waterfall_image_data[waterfall_image_data < -10] = -10

        # Apply Gaussian blur to smooth the display
        waterfall_blurred = gaussian_filter(waterfall_image_data, sigma=0.75)

        # Normalize the data to [0, 255]
        min_val = waterfall_blurred.min()
        max_val = waterfall_blurred.max()
        if max_val - min_val > 0:
            normalized = (waterfall_blurred - min_val) / (max_val - min_val)
        else:
            normalized = waterfall_blurred - min_val
        normalized = (normalized * 255).astype(np.uint8)

        # Map normalized values to colors using the LUT (vectorized lookup)
        brightness_factor = 0.5  # Adjust brightness factor as needed
        color_image = (rgb_lut[normalized] * brightness_factor).astype(np.uint8)

        # Create a pygame surface from the color image.
        surf_array = np.transpose(color_image, (1, 0, 2))
        spec_surface = pygame.surfarray.make_surface(surf_array)
        
        # Scale the spectrogram surface to fill the window
        spec_surface = pygame.transform.scale(spec_surface, (window_width, window_height))

        # Blit the spectrogram onto the screen
        screen.blit(spec_surface, (0, 0))

    # Overlay the PNG image
    screen.blit(overlay_image1, (0, 10))  # Change (100, 100) to position it differently
    
    # Overlay text on top of the spectrogram
    text1 = boldfont.render("Welcome to MAiSTRO: The Classical Music Composition Model!", True, (255, 255, 255))
    screen.blit(text1, (50, 150))

    # Overlay text on top of the spectrogram
    text2 = font.render("Start by choosing between generation mode or collaboration mode:", True, (255, 255, 255))
    screen.blit(text2, (50, 200))

    # Draw buttons with dynamic colors based on selection
    if active_button1 == "generation":
        pygame.draw.rect(screen, WHITE, button1_1)  # White background
        pygame.draw.rect(screen, WHITE, button1_1, 2)  # Border
        text1_color = BLACK  # Black text

        button_generate_alpha = fade_in(button_generate,button_generate_visible,button_generate_surface,button_generate_alpha,active_button2,"generate",button_width,button_height,"Generate")
        button_dropdown_alpha = fade_in(button_dropdown,button_dropdown_visible,button_dropdown_surface,button_dropdown_alpha,active_button2,"dropdown",button_width*2.5,button_height,option)
        button_record_alpha = fade_in(button_record,button_dropdown_visible,button_record_surface,button_record_alpha,active_button2,"record",button_width,button_height,"Record")
    else:
        pygame.draw.rect(screen, BLACK, button1_1)  # Black background
        pygame.draw.rect(screen, WHITE, button1_1, 2)  # White border
        text1_color = WHITE  # White text

    if active_button1 == "collaboration":
        if (selected_port_index is not None
            and dropdown_options[selected_port_index] != "No MIDI devices found"):

            # Draw the Record button
            pygame.draw.rect(screen, WHITE, button_record)  # White background
            pygame.draw.rect(screen, BLACK, button_record, 2)  # Black border
            text_record = buttonfont.render("Record", True, BLACK)
            screen.blit(
                text_record,
                (
                    button_record.x + (button_record.width - text_record.get_width()) // 2,
                    button_record.y + (button_record.height - text_record.get_height()) // 2
                )
            )
        
        button_dropdown_alpha = fade_in(button_dropdown,button_dropdown_visible,button_dropdown_surface,button_dropdown_alpha,active_button2,"dropdown",button_width*2.5,button_height,option)
        button_generate_alpha = fade_in(button_generate,button_generate_visible,button_generate_surface,button_generate_alpha,active_button2,"generate",button_width,button_height,"Generate")
        button_record_alpha = fade_in(button_record,button_dropdown_visible,button_record_surface,button_record_alpha,active_button2,"record",button_width,button_height,"Record")

    else:
        pygame.draw.rect(screen, BLACK, button1_2)  # Black background
        pygame.draw.rect(screen, WHITE, button1_2, 2)  # White border
        text2_color = WHITE  # White text

    # Render text
    text1 = buttonfont.render("Generation", True, text1_color)
    text2 = buttonfont.render("Collaboration", True, text2_color)
   
    text4_surface = render_fading_text("Connect your MIDI device",font,WHITE,button_dropdown_alpha)
    screen.blit(text4_surface, (50, 350))

    # Center text inside buttons
    screen.blit(text1, (button1_1.x + (button_width - text1.get_width()) // 2, button1_1.y + (button_height - text1.get_height()) // 2))
    screen.blit(text2, (button1_2.x + (button_width - text2.get_width()) // 2, button1_2.y + (button_height - text2.get_height()) // 2))

    if dropdown_active:
        draw_dropdown_menu()

    # Update the display
    pygame.display.flip()
    clock.tick(30)  # Limit to ~30 FPS

pygame.quit()
sys.exit()