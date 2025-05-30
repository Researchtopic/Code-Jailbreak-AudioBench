import os
import librosa
import soundfile as sf
from tqdm import tqdm  # For displaying progress bar

# Input and output folder paths
input_folder = 'datasets/filtered_MM-Safetybench/audio_original_2'  # Original audio folder
output_folder = 'datasets/filtered_MM-Safetybench/tone/test_2'  # Output folder for target tone

# Create folder if output folder doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define number of semitones for pitch shift
semitone_variation = +4 # Positive semitone change

# Get all .wav files in input folder
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # Ensure processing in filename order

print(f"📂 Found {len(audio_files)} audio files, generating tone variants...")

# Traverse each audio file and generate pitch-shifted files
for filename in tqdm(audio_files, desc="Processing..."):
    try:
        # Load audio file
        input_path = os.path.join(input_folder, filename)
        y, sr = librosa.load(input_path, sr=None)  # Keep original sample rate

        # Pitch shift
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitone_variation)

        # Generate new filename: audio_<number>_tone+12st.wav
        original_id = os.path.splitext(filename)[0]  # Extract original ID (e.g., 001)
        new_filename = f"{original_id}_tone+4st.wav"
        output_path = os.path.join(output_folder, new_filename)

        # Save pitch-shifted audio
        sf.write(output_path, y_shifted, sr)
    except Exception as e:
        print(f"❌ Failed to process {filename}, error: {e}")

print("✅ All audio files have been generated successfully!")