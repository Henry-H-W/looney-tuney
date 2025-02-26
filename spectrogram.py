from collections import deque
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import sys
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
import time
import threading
from scipy.ndimage import gaussian_filter
import os
import random

def generatePgColormap(cm_name):
    """Converts a matplotlib colormap to a pyqtgraph colormap."""
    colormap = plt.get_cmap(cm_name)
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0-255 for Qt
    return lut


# Constants
CHUNKSIZE = 2048
N_FFT = 16384  # Increased FFT size for higher resolution
WATERFALL_DURATION = 10  # Limit to 10 seconds
OVERLAP_RATIO = 0.5  # 50% overlap for smoother scrolling
chunk_size = int(CHUNKSIZE * (1 - OVERLAP_RATIO))  # Adjust chunk size for overlap
EPS = 1e-8

# find the most recent audio file in the directory
audio_files = [f for f in os.listdir() if f.endswith((".wav", ".mp3"))]
if not audio_files:
    raise FileNotFoundError("No audio files found in the directory.")

# Load MP3 file
audio_path = max(audio_files, key=os.path.getctime)
# Detect file format based on extension
file_ext = os.path.splitext(audio_path)[1][1:]  # Extracts "wav" or "mp3"

# Load the correct format
audio = AudioSegment.from_file(audio_path, format=file_ext).set_channels(1)  # Convert to mono

sample_rate = audio.frame_rate  # Get sample rate

# Convert to NumPy array (16-bit PCM format)
samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
samples /= np.max(np.abs(samples))  # Normalize audio

# Calculate number of spectrogram frames
WATERFALL_FRAMES = int(WATERFALL_DURATION * sample_rate / chunk_size)

# Frequency vector
FREQ_VECTOR = np.fft.rfftfreq(N_FFT, d=1 / sample_rate)
FREQ_LIMIT = 5000  # Set the max frequency to 10 kHz
freq_mask = FREQ_VECTOR <= FREQ_LIMIT  # Create a mask for filtering
FREQ_VECTOR = FREQ_VECTOR[freq_mask]  # Apply the mask to keep only 0-10kHz

# PyQtGraph Window
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget()
# win.resize(1000, 600)
win.showMaximized() # for opening in fullscreen mode
win.setWindowTitle("MP3 Spectrogram Visualizer")

# Waterfall Plot
waterfall_data = deque([np.full(len(FREQ_VECTOR), -10)] * WATERFALL_FRAMES, maxlen=WATERFALL_FRAMES)
waterfall_plot = win.addPlot(title="")
waterfall_plot.setXRange(0, WATERFALL_DURATION)
waterfall_plot.hideAxis("left")  # Hide the Y-axis
waterfall_plot.hideAxis("bottom")  # Hide the X-axis
waterfall_image = pg.ImageItem()
waterfall_plot.addItem(waterfall_image)

# List of available colormaps (feel free to add more)
COLORMAPS = [
    "viridis", "inferno", "magma", "cividis", "turbo",
    "gray", "copper", "bone", "cubehelix"
]

# Randomly select a colormap
selected_colormap = random.choice(COLORMAPS)
print(f"🎨 Selected Colormap: {selected_colormap}")  # Debugging output

lut = generatePgColormap(selected_colormap)

waterfall_image.setLookupTable(lut)
waterfall_image.setOpacity(0.5)  # Adjust transparency
waterfall_image.setAutoDownsample(True)  # Enable interpolation for smoother visuals
transform = QtGui.QTransform()
# transform.scale(chunk_size / sample_rate, FREQ_VECTOR.max() * 2. / N_FFT)
transform.scale(chunk_size / sample_rate, 10000 / len(FREQ_VECTOR))  # Scale to 10 kHz range
waterfall_image.setTransform(transform)

# Initialize playback variables
current_index = 0
start_time = None  # Track when audio starts


def update_waterfall():
    """Update the spectrogram by processing chunks from the MP3 file, keeping it in sync with audio playback."""
    global current_index, start_time

    # Sync with audio playback timing
    if start_time is None:
        return  # Don't process until playback starts
    elapsed_time = time.time() - start_time
    expected_index = int(elapsed_time * sample_rate)  # Expected sample index

    # Ensure real-time sync
    if expected_index > current_index:
        current_index = expected_index

    if current_index + chunk_size < len(samples):
        data = samples[current_index:current_index + chunk_size]
        current_index += chunk_size
    else:
        return  # Stop updating when the file is fully processed

    # Compute FFT
    # X = np.abs(np.fft.rfft(np.hanning(data.size) * data, n=N_FFT))
    X = np.abs(np.fft.rfft(np.hanning(data.size) * data, n=N_FFT))[freq_mask]  # Apply the mask
    new_frame = np.log10(X + 1e-12)

    # Maintain left-to-right scrolling (keep original behavior)
    waterfall_data.appendleft(new_frame)  # Insert new frame at the start

    arr = np.array(waterfall_data)

    # Remove very low values (background noise) before blurring
    arr[arr < -10] = -10  # Forces weak signals to be pure black

    # Apply Gaussian blur to reduce pixelation
    arr = gaussian_filter(arr, sigma=1)

    if arr.size > 0:
        waterfall_image.setImage(arr, levels=(arr.min(), arr.max()))


def play_audio():
    global start_time
    start_time = time.time()  # Mark playback start time
    play(audio)  # Play MP3 file using pydub


# Start the audio in a separate thread
audio_thread = threading.Thread(target=play_audio, daemon=True)
audio_thread.start()

# Timer for updating spectrogram
timer_waterfall = QtCore.QTimer()
timer_waterfall.timeout.connect(update_waterfall)
timer_waterfall.start(max(1, int(1000 * CHUNKSIZE / sample_rate * 0.5)))  # Speed up slightly
win.show()
sys.exit(app.exec_())