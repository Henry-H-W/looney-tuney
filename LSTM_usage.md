## Usage
- Fill the 'data' folder with the MIDI files you want to train the model on.
- To train your model, use the file 'lstm.ipynb'. After all MIDI files have been processed, a file called "notes" will be created in the 'data' folder; this needs to be deleted each time you want to use **new** data to train the model. After each epoch, a checkpoint will be used to save the model progress up to that point in the 'model' folder (probably as "best_model.keras").
- To create a new song after training, use the file 'generate.ipynb'.

## Installation
Make sure to install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```