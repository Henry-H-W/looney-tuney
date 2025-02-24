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
Ensure that you have your MIDI dataset inside the `dataset/` directory. The `main.py` script will parse and preprocess these files. If some of your midi files contain multiple tracks, `midi_processing.py` can help condense everything into one track.

### 2. Train the Model
Before running the model, make sure you have access to a GPU! CPU training is heavily discouraged due to time and resource-intensity (we recommend Google Colab for free GPU use). To train the LSTM model on the provided dataset, run:

```bash
python main.py
```

This will:
1. Parse MIDI files and extract notes.
2. Convert notes into sequences for training.
3. Train an LSTM model using the prepared sequences.
4. Save model weights after training. 

### 3. Generate Music
Once the model is trained, comment the training code. You can then generate new MIDI compositions using:

```bash
python main.py
```

The script will:
1. Load the trained model.
2. Generate a sequence of musical notes.
3. Convert the generated sequence into a MIDI file.
4. Save the output as `test_output.mid`.

### 4. Playing the Generated MIDI File
To play the generated music and save it as an audible file, run:
```bash
python play.py
```

## Notes
- The model uses a Bi-directional LSTM architecture with self-attention to generate more coherent musical sequences.
- The generated output is saved as `generated_music_[date]_[timestamp].mid`.
- Pre-trained model weights should be available in `weights-epoch[epoch#]-[loss].keras` to generate meaningful results without retraining.
- We found that 200+ MIDI files provide the best results (a good sign is to check the size of the resulting 'notes' file: if it's 1-2Mb, you have a good amount of data!)
- We found training for 30 epochs on a 200-file dataset works best, but this number can be different based on the dataset.
- We found that a loss around 0.2000 - 0.4000 is a good balance of structure & variation without overfitting.
- A Jupyter Notebook version of the code is also inside this repository for debugging and more convenient code execution.
- If you're worried about overfitting, run your dataset and generated midis through `overfit_check.py` which determines if the note distributions of your generated midi match any tracks from the dataset.

## License
This project is open-source and can be modified or distributed under the MIT License.