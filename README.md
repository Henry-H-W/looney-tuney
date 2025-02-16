# LSTM Music Composition Model

This project is a deep learning-based music composition system using Long Short-Term Memory (LSTM) neural networks. The model learns musical patterns from classical MIDI files and generates new music based on the learned sequences. This project is associated with Western Cyber Society and contributed by:
- Henry Wang
- Richard Augustine
- Shawn Yuen
- Elbert Chao
- Ryan Huang
- David Lim
- Raymond Li
- Leo Karras
- Sahil Patel

## Prerequisites
Ensure you have the necessary dependencies installed before running the project. You can install them using:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `os`
- `glob`
- `pickle`
- `music21`
- `keras`
- `tensorflow`
- `keras-self-attention`

## Getting Started
### 1. Prepare the Dataset
Ensure that you have MIDI files inside the `full_set_beethoven_mozart/` directory. The script will parse and preprocess these files.

### 2. Train the Model
Before running the model, make sure you have access to a GPU! CPU training is heavily discouraged due to time and resource-intensity. To train the LSTM model on the provided dataset, run:

```bash
python main.py
```

This will:
1. Parse MIDI files and extract notes.
2. Convert notes into sequences for training.
3. Train an LSTM model using the prepared sequences.
4. Save model weights after training.

### 3. Generate Music
Once the model is trained, you can generate new MIDI compositions using:

```bash
python main.py
```

The script will:
1. Load the trained model.
2. Generate a sequence of musical notes.
3. Convert the generated sequence into a MIDI file.
4. Save the output as `test_output.mid`.

### 4. Playing the Generated MIDI File
To play the generated music, use a MIDI player such as:
- [MuseScore](https://musescore.org/)
- [VLC Media Player](https://www.videolan.org/vlc/)
- Any DAW (Digital Audio Workstation) like FL Studio or Ableton

## Notes
- The model uses a Bi-directional LSTM architecture with self-attention to generate more coherent musical sequences.
- The generated output is saved in `test_output.mid`.
- Pre-trained model weights should be available in `weights-1LSTMAtt1LSTMLayer-030-0.3911.keras` to generate meaningful results without retraining.

## License
This project is open-source and can be modified or distributed under the MIT License.


