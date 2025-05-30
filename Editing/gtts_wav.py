import pandas as pd
import os
import time
from gtts import gTTS
from pydub import AudioSegment

# 📌 Input CSV file path
input_csv_path = "Datasets_Text/advbench_armful_behaviors.csv"

# 📌 Output audio file directory (ensure WAV format)
output_audio_dir = "datasets/filtered_datasets/filtered_advbench/audio_original"
os.makedirs(output_audio_dir, exist_ok=True)  # Ensure directory exists

# 📌 Read CSV file
df = pd.read_csv(input_csv_path)

# ✅ Ensure CSV contains 'question' column
if "question" not in df.columns:
    raise ValueError("Error: No 'question' column in CSV file!")

# 📌 Set range for generated audio
start_index = 0 
end_index = len(df)  # Avoid exceeding data range

# 🎤 Traverse data and generate WAV audio
for index in range(start_index, end_index):
    row = df.iloc[index]
    question = row["question"].strip()

    if not isinstance(question, str) or question == "":
        print(f"⚠️ Skipping blank text (row {index + 1})")
        continue

    # Filename format: audio_001.wav, audio_002.wav, ...
    audio_filename = f"audio_{index + 1:03d}.mp3"  
    mp3_filepath = os.path.join(output_audio_dir, audio_filename)
    wav_filepath = mp3_filepath.replace(".mp3", ".wav")

    print(f"🎧 Generating audio: {wav_filepath}")

    try:
        # 🔹 Generate MP3 using gTTS
        tts = gTTS(text=question, lang='en')
        tts.save(mp3_filepath)

        # 🔹 Convert MP3 to WAV
        audio = AudioSegment.from_mp3(mp3_filepath)
        audio.export(wav_filepath, format="wav")

        # 🔹 Delete original MP3 (optional)
        os.remove(mp3_filepath)

        time.sleep(1)  # Avoid API rate limiting
    except Exception as e:
        print(f"❌ Failed to generate audio (row {index + 1}): {e}")
        continue

print(f"\n✅ Successfully generated {end_index - start_index} WAV audio files, stored in {output_audio_dir}")