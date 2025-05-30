import os
import librosa
import soundfile as sf

# Input and output folder paths
input_folder = 'strict_small_screams/audio'  # Input folder path
output_folder = 'dataset_grid/emotion/screams/x5_x1.5_white_noise'  # Output folder path

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Traverse all audio files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        filepath = os.path.join(input_folder, filename)
        
        # Load audio file
        y, sr = librosa.load(filepath)
        
        # Define emphasized time segment (first 1 second)
        start_time = 0.0
        end_time = 1.0
        
        # Convert time to sample points
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Increase volume
        gain = 5.0  # Multiplication factor
        y[start_sample:end_sample] *= gain
        
        # Generate new filename
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_emphasis_x5{ext}"
        
        # Save processed audio
        output_path = os.path.join(output_folder, new_filename)
        sf.write(output_path, y, sr)
        print(f"Processed and saved: {output_path}")


print("All audio files have been processed.")