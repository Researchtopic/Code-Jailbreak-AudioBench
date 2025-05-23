import os
import subprocess

# Original audio directory (only contains original .wav files)
input_folder = 'datasets_supplement/dataset_grid/emotion/screams/x5_x0.5_white_noise'

# Output directories
output_folders = {
    0.5: 'datasets_supplement/dataset_grid/emotion/screams/x5_x0.5_white_noise',
    1.5: 'datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise'
}

# Ensure output folders exist
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# Get all original .wav files (exclude already processed ones)
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav') and "speedx" not in f]
audio_files.sort()

speed_rates = [0.5, 1.5]

# Process audio and save speed-changed versions
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)

    for speed_rate in speed_rates:
        original_id = os.path.splitext(audio_file)[0]
        new_filename = f"{original_id}_speedx{speed_rate}.wav"
        output_path = os.path.join(output_folders[speed_rate], new_filename)

        cmd = ["sox", input_path, output_path, "tempo", "-s", str(speed_rate)]
        subprocess.run(cmd, check=True)
        print(f'Processed {audio_file} -> {new_filename} (Speed x{speed_rate})')

# Delete all non-speedx original audio files in all directories
for folder in output_folders.values():
    for f in os.listdir(folder):
        if f.endswith('.wav') and "speedx" not in f:
            os.remove(os.path.join(folder, f))
            print(f"Deleted original audio: {f} from {folder}")

print("All audio files have been processed and original files deleted.")