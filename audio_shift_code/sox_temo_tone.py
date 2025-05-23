import os
import subprocess
from tqdm import tqdm  # For displaying progress bar

# Input audio folder path
input_folder = "datasets/filtered_datasets/filtered_advbench/audio_original"

# Define output folders, organized by pitch level
output_base = "datasets/filtered_datasets/filtered_advbench/tone"
tone_variations = {
    "+4st": os.path.join(output_base, "+4"),
    "-4st": os.path.join(output_base, "-4"),
    "+8st": os.path.join(output_base, "+8"),
    "-8st": os.path.join(output_base, "-8"),
}

# Ensure all output folders exist
for folder in tone_variations.values():
    os.makedirs(folder, exist_ok=True)

# Get all .wav files
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()

print(f"📂 Found {len(audio_files)} audio files, performing global pitch shift using SOX...")

# Traverse each audio file and generate all pitch variations
for i, filename in tqdm(enumerate(audio_files, start=1), desc="Processing..."):
    try:
        input_path = os.path.join(input_folder, filename)
        original_id = f"audio_{i:03d}"  # Change ID to 3-digit format, e.g., audio_001

        for tone, output_folder in tone_variations.items():
            # Calculate sox pitch value
            semitone = int(tone.replace("st", "").replace("+", ""))  # Extract numeric value
            pitch_shift = semitone * 100  # Sox uses cents (1/100 semitone) as unit

            # Generate output file path
            new_filename = f"{original_id}_tone{tone}.wav"
            output_path = os.path.join(output_folder, new_filename)

            # Call sox for pitch shifting
            cmd = ["sox", input_path, output_path, "pitch", str(pitch_shift)]
            subprocess.run(cmd, check=True)

        print(f"✅ Completed processing {filename}")

    except Exception as e:
        print(f"❌ Failed to process {filename}, error: {e}")

print("🎉 ✅ All audio files have been globally pitch-shifted and saved to corresponding folders!")