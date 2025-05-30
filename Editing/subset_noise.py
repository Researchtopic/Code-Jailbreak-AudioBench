import os
from pydub import AudioSegment

# Input paths
original_audio_dir = "datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_machine"
noise_audio_path = "noise_original/noise_original_change/machine_noise_normalized_noise.wav"
output_audio_dir = "datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_machine"

# Ensure output directory exists
os.makedirs(output_audio_dir, exist_ok=True)

# Load noise audio
noise = AudioSegment.from_file(noise_audio_path)

# Current noise volume
current_volume = noise.dBFS

# Target noise volume
target_volume = -30.0

# Calculate required gain
gain = target_volume - current_volume

# Adjust noise volume
adjusted_noise = noise.apply_gain(gain)

print(f"Current Noise Volume: {current_volume:.2f} dBFS")
print(f"Target Noise Volume: {target_volume:.2f} dBFS")
print(f"Applied Gain: {gain:.2f} dB")

# Function: repeat noise to match original audio length
def match_noise_length(original, noise):
    if len(noise) >= len(original):
        return noise[:len(original)]
    else:
        repeats = len(original) // len(noise) + 1
        extended_noise = noise * repeats
        return extended_noise[:len(original)]

# Traverse original audio files
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav") and "_machine" not in filename:
        original_audio_path = os.path.join(original_audio_dir, filename)
        output_audio_path = os.path.join(output_audio_dir, filename.replace(".wav", "_machine.wav"))
        
        try:
            original_audio = AudioSegment.from_file(original_audio_path)
            noise_to_add = match_noise_length(original_audio, adjusted_noise)
            mixed_audio = original_audio.overlay(noise_to_add)
            mixed_audio.export(output_audio_path, format="wav")

            # Delete original audio file
            os.remove(original_audio_path)

            print(f"Processed and deleted: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"\nFinal Noise Volume Added: {target_volume:.2f} dBFS")