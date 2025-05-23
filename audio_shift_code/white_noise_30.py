import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

'''
cd datasets/filtered_MM-Safetybench/audio_original

# Scan all MP3 disguised as WAV files and convert to true WAV
for file in *.wav; do
    if file "$file" | grep -q "MPEG ADTS"; then
        echo "⚠️ $file is actually an MP3. Converting to true WAV..."
        ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 -y "${file}.fixed.wav"
        mv "${file}.fixed.wav" "$file"  # Overwrite original file
    fi
done

echo "✅ All MP3 files have been converted to WAV!"
'''

# Define folder paths
original_audio_folder = "datasets/filtered_datasets/filtered_advbench/audio_original"
output_folder = "datasets/filtered_datasets/filtered_advbench/noise/white_noise"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Set white noise volume
white_noise_gain_dB = -30.0

# Traverse all audio files
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)

        try:
            # Try to load original audio directly
            original_audio = AudioSegment.from_wav(original_audio_path)
        except:
            # If loading fails, convert format and overwrite original file
            print(f"⚠️ Warning: {filename} format issue detected, converting to standard WAV...")

            # Use ffmpeg to convert audio format (overwrite original file)
            os.system(f"ffmpeg -i \"{original_audio_path}\" -acodec pcm_s16le -ar 44100 -y \"{original_audio_path}\"")

            # Try to load converted audio again
            original_audio = AudioSegment.from_wav(original_audio_path)

        # Generate white noise with same length as original audio
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))

        # Adjust white noise volume
        adjusted_white_noise = white_noise.apply_gain(white_noise_gain_dB)

        # Overlay white noise
        combined_audio = original_audio.overlay(adjusted_white_noise)

        # Generate new filename (keep original audio name, only add suffix)
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        output_path = os.path.join(output_folder, new_filename)

        # Export audio
        combined_audio.export(output_path, format="wav")

        print(f"✅ Processed {filename} -> {new_filename}")

print("All files processed with added white noise!")