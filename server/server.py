from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from generate import generate_music_from_midi
import os

UPLOAD_FOLDER = os.path.abspath("uploads")  # Ensures it's in the root project directory
OUTPUT_FILE = os.path.abspath("output.mid")  # Also place output file in root if needed

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__, static_folder=OUTPUT_FOLDER)
CORS(app)  # Allow frontend requests
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles MIDI file upload and processing"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"✅ File saved: {file_path}")

    try:
        # Process the uploaded MIDI file
        generate_music_from_midi(file_path, OUTPUT_FILE)
        print(f"✅ MIDI generated successfully: {OUTPUT_FILE}")
        return jsonify({"message": "File processed successfully", "file_url": "/output.mid"}), 200
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return jsonify({"error": "Failed to process MIDI file", "details": str(e)}), 500

@app.route('/output.mid', methods=['GET'])
def serve_output_midi():
    """Serve the generated MIDI file"""
    if os.path.exists(OUTPUT_FILE):
        return send_from_directory(OUTPUT_FOLDER, "output.mid", as_attachment=False)
    else:
        return jsonify({"error": "Generated MIDI file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
