# MAiSTRO: The Classical Music Composition Model

This project is a deep learning-based music composition system using Long Short-Term Memory (LSTM) neural networks. The model learns musical patterns from MIDI files and generates new music based on the learned sequences. This project is associated with Western Cyber Society and contributed by:
- Henry Wang
- Richard Augustine
- Shawn Yuen
- Elbert Chao
- Ryan Huang
- David Lim
- Raymond Li
- Leo Karras

## Prerequisites
Ensure you have the necessary dependencies installed before running the project. You can install them using:

```bash
pip install -r requirements.txt
```

## Getting Started: Training the Model
### 1. Prepare the Dataset
Ensure that you have your MIDI dataset inside the `dataset/` directory (make sure the files end with .mid). If some of your midi files contain multiple tracks / instruments, `midi_dataset_processing.py` can help condense everything into one track.

### 2. Train the Model
To train the LSTM model on the provided dataset, run:

```bash
python train.py
```

This will:
1. Parse MIDI files and extract notes to be contained in the `data/` directory. Make sure to delete / replace this file everytime you want to train the model on a new dataset!
2. Convert notes into sequences for training.
3. Train an LSTM model using the prepared sequences.
4. Save model weights after training with the format `weights-epoch[epoch#]-[loss].keras`. 

### 3. Generate Music
Once the model is trained, you can generate new MIDI compositions using:

```bash
python generate.py
```

This will:
1. Load the trained model.
2. Generate a sequence of musical notes.
3. Convert the generated sequence into a MIDI file.
4. Save the output as `generated_music_[date]_[timestamp].mid`.

### 4. Playing the Generated MIDI File
To play the generated music (using instruments from the `soundfonts/` directory) and save it as an audible file, run:
```bash
python midi_to_audio.py
```

## Exploring Further: Applying the Model
Now that we have a trained model, we can use it as a tool to help musicians compose by completing melodies from user-inputted midi! Through our Python GUI, you can achieve this by running:
```bash
python audio_to_spectrogram.py
```
Which uses the following scripts:
- `keyboard_to_midi.py` to take a keyboard's input and turn it into a midi file.
- `collaborate.py` to generate midi and effectively extend the inputted midi recording.
- `midi_to_audio.py` to convert the new midi into audio.

## Notes & Troubleshooting
- Before running the model, make sure you have access to a GPU! CPU training is heavily discouraged due to time and resource-intensity (we recommend Google Colab for free GPU use, Jupyter Notebook versions of train.py and generate.py are provided in this repository).
- We found that 200+ MIDI files provide the best results (a good sign is to check the size of the resulting 'notes' file: if it's 1-2Mb, you have a good amount of data!)
- We found training for 30 epochs on a 200-file dataset works best, but this number can be different based on the dataset.
- We found that a loss around 0.2 - 0.4 is a good balance of structure & variation without overfitting.
- If you're worried about overfitting, run your dataset and generated midis through `overfit_check.py` which determines if the note distributions of your generated midi match any tracks from the dataset.
- The `python-rtmidi` library needs <a href="https://visualstudio.microsoft.com/visual-cpp-build-tools/">Microsoft C++ Build Tools</a> installed to run 
- `keyboard_to_midi.py` takes a keyboard's input and turns it into a midi file for demonstration purposes. Make sure you have no applications that may use the MIDI input open (such as DAWs) so that the script has no problem receiving the MIDI information from your device.
- `midi_to_audio.py` records the audible file in 120bpm. This means, if the input midi is intended to be played with another bpm, the playbck may not sound correct.
- `audio_to_spectrogram.py` produces a spectrogram of any wav or mp3 file.
- `audio_to_spectrogram.py` has GUI components that use the <a href="https://fonts.google.com/specimen/Lato">Lato Font Family</a>. Please install it or replace the font with one that is available on your device.

## References
- <a href="https://medium.com/@alexissa122/generating-original-classical-music-with-an-lstm-neural-network-and-attention-abf03f9ddcb4">Generating Original Classical Music with an LSTM Neural Network and Attention</a>
- <a href="https://youtu.be/zpZDwqsgSpc?si=LaH-QSm2hTHLAf8E">Programming with MIDI in Python | Responding to MIDI Messages</a>
- <a href="https://swharden.com/blog/2010-06-19-simple-python-spectrograph-with-pygame/">Simple Python Spectrograph with PyGame</a>

## License
This project is open-source and can be modified or distributed under the MIT License.