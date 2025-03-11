from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from generate import generate_music_from_midi
import os

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # Store uploaded MIDI files
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")  # Store generated MIDI files
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "generated.mid")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # Allow frontend requests
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200

@app.route('/generate', methods=['POST'])
def generate():
    """Generate music using the AI model"""
    data = request.get_json()
    if not data or "filename" not in data:
        return jsonify({"error": "No filename provided"}), 400

    filename = data["filename"]
    input_midi_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(input_midi_path):
        return jsonify({"error": f"MIDI file '{filename}' not found"}), 404

    try:
        generate_music_from_midi(input_midi_path, OUTPUT_FILE)
        print(f"✅ MIDI generated successfully: {OUTPUT_FILE}")
        return jsonify({"message": "Music generated successfully", "generated_filename": "generated.mid"}), 200
    except Exception as e:
        print(f"❌ Error generating music: {str(e)}")
        return jsonify({"error": "Failed to generate music", "details": str(e)}), 500

@app.route('/output/<filename>', methods=['GET'])
def serve_output_midi(filename):
    """Serve the generated MIDI file"""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)
    else:
        return jsonify({"error": "Generated MIDI file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)