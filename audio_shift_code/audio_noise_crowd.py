import os
from pydub import AudioSegment

# Input paths
original_audio_dir = "accent/black/x2_x0.5_machine"
noise_audio_path = "noise_original/machine_noise_normalized_noise.wav"
output_audio_dir = "accent/black/x2_x0.5_machine"

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

# Print adjustment information
print(f"Current Noise Volume: {current_volume:.2f} dBFS")
print(f"Target Noise Volume: {target_volume:.2f} dBFS")
print(f"Applied Gain: {gain:.2f} dB")

# Function: repeat noise to match original audio length
def match_noise_length(original, noise):
    if len(noise) >= len(original):
        return noise[:len(original)]  # Noise is long enough, crop it
    else:
        repeats = len(original) // len(noise) + 1  # Calculate required repetitions
        extended_noise = noise * repeats  # Repeat noise
        return extended_noise[:len(original)]  # Crop to exact length

# Traverse original audio files
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_dir, filename)
        output_audio_path = os.path.join(output_audio_dir, filename.replace(".wav", "_noise_crowd.wav"))
        
        try:
            # Load original audio
            original_audio = AudioSegment.from_file(original_audio_path)
            
            # Adjust noise length to match original audio
            noise_to_add = match_noise_length(original_audio, adjusted_noise)
            
            # Mix audio
            mixed_audio = original_audio.overlay(noise_to_add)
            
            # Export audio with noise
            mixed_audio.export(output_audio_path, format="wav")
            print(f"Processed {filename} -> {output_audio_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Finally print the adjusted noise volume
print(f"\nFinal Noise Volume Added: {target_volume:.2f} dBFS")