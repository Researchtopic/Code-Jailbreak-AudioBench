import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# Define folder paths
original_audio_folder = "audio_original"
output_folder = "noise/white_noise"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Traverse all audio files in the folder
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)
        
        # Load original audio file
        original_audio = AudioSegment.from_wav(original_audio_path)
        
        # Get original audio loudness (dBFS)
        original_loudness = original_audio.dBFS
        
        # Generate white noise with the same length as original audio
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))
        
        # Adjust white noise loudness to match original audio
        change_in_dBFS = original_loudness - white_noise.dBFS
        adjusted_white_noise = white_noise.apply_gain(change_in_dBFS)
        
        # Mix adjusted white noise with original audio
        combined_audio = original_audio.overlay(adjusted_white_noise)
        
        # Generate new filename
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        
        # Export mixed audio file to output folder
        output_path = os.path.join(output_folder, new_filename)
        combined_audio.export(output_path, format="wav")
        
        print(f"Processed {filename} -> {new_filename}")

print("✅ All files processed with added white noise!")