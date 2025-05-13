import pandas as pd
import os
import time
from gtts import gTTS
from pydub import AudioSegment

# 📌 输入 CSV 文件路径
input_csv_path = "/home/ubuntu/Jing/Qwen-Audio/Datasets_Text/advbench_armful_behaviors.csv"

# 📌 输出音频文件目录（确保 WAV 格式）
output_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/audio_original"
os.makedirs(output_audio_dir, exist_ok=True)  # 确保目录存在

# 📌 读取 CSV 文件
df = pd.read_csv(input_csv_path)

# ✅ 确保 CSV 包含 'question' 列
if "question" not in df.columns:
    raise ValueError("Error: CSV 文件中没有 'question' 列!")

# 📌 设置生成的音频范围
start_index = 0 
end_index = len(df)  # 避免超出数据范围

# 🎤 遍历数据并生成 WAV 音频
for index in range(start_index, end_index):
    row = df.iloc[index]
    question = row["question"].strip()

    if not isinstance(question, str) or question == "":
        print(f"⚠️ 跳过空白文本 (行 {index + 1})")
        continue

    # 文件名格式：audio_001.wav, audio_002.wav, ...
    audio_filename = f"audio_{index + 1:03d}.mp3"  
    mp3_filepath = os.path.join(output_audio_dir, audio_filename)
    wav_filepath = mp3_filepath.replace(".mp3", ".wav")

    print(f"🎧 生成音频: {wav_filepath}")

    try:
        # 🔹 使用 gTTS 生成 MP3
        tts = gTTS(text=question, lang='en')
        tts.save(mp3_filepath)

        # 🔹 转换 MP3 为 WAV
        audio = AudioSegment.from_mp3(mp3_filepath)
        audio.export(wav_filepath, format="wav")

        # 🔹 删除原始 MP3（可选）
        os.remove(mp3_filepath)

        time.sleep(1)  # 避免 API 速率限制
    except Exception as e:
        print(f"❌ 生成音频失败 (行 {index + 1}): {e}")
        continue

print(f"\n✅ 成功生成 {end_index - start_index} 个 WAV 音频文件，存储在 {output_audio_dir}")