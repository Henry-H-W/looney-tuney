import os
import sys
import time
import random
import threading
from collections import deque
import cmasher as cmr  # Import cmasher for additional colormaps

import numpy as np
import pygame
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from scipy.ndimage import gaussian_filter

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

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)

# ----- Find and load the most recent audio file -----
audio_files = [f for f in os.listdir() if f.endswith((".wav", ".mp3"))]
if not audio_files:
    raise FileNotFoundError("No audio files found in the directory.")

audio_path = max(audio_files, key=os.path.getctime)
file_ext = os.path.splitext(audio_path)[1][1:]  # e.g., "wav" or "mp3"
print(f"Loading audio file: {audio_path}")

audio = AudioSegment.from_file(audio_path, format=file_ext).set_channels(1)  # Convert to mono
sample_rate = audio.frame_rate

# Convert audio samples to a normalized numpy float32 array
samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
if np.max(np.abs(samples)) > 0:
    samples /= np.max(np.abs(samples))

# ----- Compute Waterfall Dimensions -----
# Number of waterfall frames (time resolution)
WATERFALL_FRAMES = int(WATERFALL_DURATION * sample_rate / chunk_size)
# Frequency vector (only keep frequencies up to FREQ_LIMIT)
FREQ_LIMIT = 4000  # Hz (change as needed)
FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=1/sample_rate)
freq_mask = FREQ_VECTOR <= FREQ_LIMIT
FREQ_VECTOR = FREQ_VECTOR[freq_mask]
num_freq_bins = len(FREQ_VECTOR)

# ----- Initialize the Waterfall Image Data -----
# We create a 2D numpy array with shape (num_freq_bins, WATERFALL_FRAMES)
# Each column will hold the log-magnitude spectrum from one FFT frame.
waterfall_image_data = np.full((num_freq_bins, WATERFALL_FRAMES), -10, dtype=np.float32)

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
# Track which button is active
active_button1 = None  # Can be "generation" or "collaboration"

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
current_index = 0  # current sample index into the audio file
start_time = None  # will be set when playback starts

def play_audio():
    global start_time
    start_time = time.time()  # mark playback start time
    play(audio)  # this call blocks; run in a separate thread

# Start audio playback in a separate thread
audio_thread = threading.Thread(target=play_audio, daemon=True)
audio_thread.start()

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
            elif button1_2.collidepoint(event.pos):
                print("Collaboration button clicked!")
                active_button1 = "collaboration"

    # Only update waterfall if playback has started
    if start_time is not None:
        elapsed_time = time.time() - start_time
        samples_played = int(elapsed_time * audio.frame_rate)  # Actual samples played
        current_index = min(samples_played, len(samples) - chunk_size)  # Ensure we don’t go out of bounds

        if current_index + chunk_size < len(samples):
            # Get the next chunk from the audio samples
            data = samples[current_index: current_index + chunk_size]
            current_index += chunk_size

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
            # The result is an array of shape (num_freq_bins, WATERFALL_FRAMES, 3)
            # Adjust brightness by scaling the RGB values (lower values = darker)
            brightness_factor = 0.5  # Adjust between 0 (black) and 1 (full brightness)
            color_image = (rgb_lut[normalized] * brightness_factor).astype(np.uint8)

            #color_image = rgb_lut[normalized]

            # Create a pygame surface from the color image.
            # Note: pygame.surfarray.make_surface expects an array with shape (width, height, 3),
            # so we transpose the array (time axis becomes x, frequency axis becomes y)
            surf_array = np.transpose(color_image, (1, 0, 2))
            spec_surface = pygame.surfarray.make_surface(surf_array)
            
            # Scale the spectrogram surface to fill the window
            spec_surface = pygame.transform.scale(spec_surface, (window_width, window_height))

            # Blit the spectrogram onto the screen
            screen.blit(spec_surface, (0, 0))
        else:
            # End of audio: optionally, you might break out or freeze the display.
            pass

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
    else:
        pygame.draw.rect(screen, BLACK, button1_1)  # Black background
        pygame.draw.rect(screen, WHITE, button1_1, 2)  # White border
        text1_color = WHITE  # White text

    if active_button1 == "collaboration":
        pygame.draw.rect(screen, WHITE, button1_2)  # White background
        pygame.draw.rect(screen, WHITE, button1_2, 2)  # Border
        text2_color = BLACK  # Black text
    else:
        pygame.draw.rect(screen, BLACK, button1_2)  # Black background
        pygame.draw.rect(screen, WHITE, button1_2, 2)  # White border
        text2_color = WHITE  # White text
        
    # Render text
    text1 = buttonfont.render("Generation", True, text1_color)
    text2 = buttonfont.render("Collaboration", True, text2_color)

    # Center text inside buttons
    screen.blit(text1, (button1_1.x + (button_width - text1.get_width()) // 2, button1_1.y + (button_height - text1.get_height()) // 2))
    screen.blit(text2, (button1_2.x + (button_width - text2.get_width()) // 2, button1_2.y + (button_height - text2.get_height()) // 2))

    # Update the display
    pygame.display.flip()
    clock.tick(30)  # Limit to ~30 FPS

pygame.quit()
sys.exit()