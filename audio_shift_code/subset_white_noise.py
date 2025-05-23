import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# Input original audio folder
original_audio_folder = "datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise"
# Output folder for audio with white noise
output_folder = "datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Set white noise target volume (unit: dBFS)
white_noise_gain_dB = -30.0

# Traverse all .wav files
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)

        try:
            original_audio = AudioSegment.from_wav(original_audio_path)
        except:
            print(f"⚠️ Warning: {filename} format issue detected, converting to standard WAV...")
            os.system(f"ffmpeg -i \"{original_audio_path}\" -acodec pcm_s16le -ar 44100 -y \"{original_audio_path}\"")
            original_audio = AudioSegment.from_wav(original_audio_path)

        # Generate white noise with same length as original audio
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))
        adjusted_white_noise = white_noise.apply_gain(white_noise_gain_dB)

        # Overlay white noise
        combined_audio = original_audio.overlay(adjusted_white_noise)

        # Name new file
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        output_path = os.path.join(output_folder, new_filename)

        # Export audio
        combined_audio.export(output_path, format="wav")

        # Delete original audio file
        os.remove(original_audio_path)

        print(f"✅ Processed {filename} -> {new_filename} and deleted original.")

print("All files processed with added white noise and originals deleted.")