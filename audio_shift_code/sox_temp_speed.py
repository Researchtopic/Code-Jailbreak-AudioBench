import os
import subprocess

# Set input audio folder path
input_folder = 'datasets/dataset_grid/accent/black/x2_x0.5_machine'

# Set output folders for different speed rates
output_folders = {
    0.5: 'datasets/dataset_grid/accent/black/x2_x0.5_machine',
    1.5: 'datasets/dataset_grid/accent/black/x2_x1.5_machine'
}

# Ensure output folders exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Get all .wav audio files
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # Sort by filename to ensure consistent processing order

# Speed rates to change
speed_rates = [0.5, 1.5]

# Process each audio file
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)

    for speed_rate in speed_rates:
        # Extract original ID (e.g., extract 001 from audio_001.wav)
        original_id = os.path.splitext(audio_file)[0]
        new_filename = f"{original_id}_speedx{speed_rate}.wav"

        # Generate output file path
        output_path = os.path.join(output_folders[speed_rate], new_filename)

        # Use sox for speed change while maintaining pitch
        cmd = ["sox", input_path, output_path, "tempo", "-s", str(speed_rate)]
        subprocess.run(cmd, check=True)

        print(f'Processed {audio_file} -> {new_filename} (Speed x{speed_rate})')

print(" All audio files have been processed using SOX (tempo).")