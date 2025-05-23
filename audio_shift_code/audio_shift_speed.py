import os
import librosa
import soundfile as sf

# Set audio folder paths
input_folder = 'datasets/filtered_MM-Safetybench/audio_original'
output_folder = 'datasets/filtered_MM-Safetybench/speed/x1.5'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all audio files in folder
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # Sort by filename to ensure consistent processing order

# Define speech speed rate
speed_rate = 1.5

# Process each audio file
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)
    
    # Load audio file
    y, sr = librosa.load(input_path)
    
    # Change speech speed
    y_changed = librosa.effects.time_stretch(y, rate=speed_rate)
    
    # Extract original ID (e.g., extract 001 from audio_001.wav)
    original_id = os.path.splitext(audio_file)[0]  # Remove extension
    new_filename = f"{original_id}_speedx{speed_rate}.wav"  # Generate filename following suggested naming convention
    
    # Generate output file path
    output_path = os.path.join(output_folder, new_filename)
    
    # Save processed audio file
    sf.write(output_path, y_changed, sr)
    
    print(f'Processed {audio_file} -> {new_filename}')

print("✅ All audio files have been processed with new naming conventions.")