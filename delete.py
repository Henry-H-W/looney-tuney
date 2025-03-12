import os
import glob

def delete_generated_files():
    """Deletes specific generated files based on the ignore list."""
    files_to_delete = [
        "output.mid",
        "recording.mid",
        "generated_output.mid",
        "output.wav",
        "extended_output.mid",
        "temp_output.wav"
    ]

    # Handle wildcard pattern
    wildcard_patterns = ["generated_music_*.mid"]

    # Delete explicitly listed files
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")

    # Delete files matching wildcard patterns
    for pattern in wildcard_patterns:
        for file in glob.glob(pattern):
            os.remove(file)
            print(f"Deleted: {file}")
