import pandas as pd
import os
from gtts import gTTS
import time

# Input CSV file path
input_csv_path = "Datasets_Text/filtered_safebench.csv"
# Output audio file directory
output_audio_dir = "datasets/filtered_safebench/audio_original"

# Ensure output directory exists
os.makedirs(output_audio_dir, exist_ok=True)

# Read CSV file
df = pd.read_csv(input_csv_path)

# Check if "question" column exists
if "question" not in df.columns:
    raise ValueError("The column 'question' does not exist in the CSV file.")

# Set range for generated audio
start_index = 0  # Start from row 1
end_index = 301  # End at row 741

# Traverse data within range and generate audio
for index in range(start_index, end_index):
    row = df.iloc[index]
    question = row["question"]
    audio_filename = f"audio_{index + 1:03d}.wav"  # Name audio files by sequence number
    audio_filepath = os.path.join(output_audio_dir, audio_filename)

    print(f"Generating audio file: {audio_filename}")
    try:
        # Generate audio using gTTS
        tts = gTTS(text=question, lang='en')
        tts.save(audio_filepath)
        time.sleep(1)  # Add delay to prevent errors from too many requests
    except Exception as e:
        print(f"Error generating audio for row {index + 1}: {e}")
        continue

print(f"Audio files from 1 to {end_index} have been successfully generated and saved to: {output_audio_dir}")