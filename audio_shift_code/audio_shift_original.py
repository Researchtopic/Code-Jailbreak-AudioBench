import pandas as pd
import os
from gtts import gTTS
import time

# 输入CSV文件路径
input_csv_path = "/home/ubuntu/Jing/Qwen-Audio/Datasets_Text/filtered_safebench.csv"
# 输出音频文件目录
output_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_safebench/audio_original"

# 确保输出目录存在
os.makedirs(output_audio_dir, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(input_csv_path)

# 检查是否包含 "question" 列
if "question" not in df.columns:
    raise ValueError("The column 'question' does not exist in the CSV file.")

# 设置生成的音频范围
start_index = 0  # 从第1行开始
end_index = 301  # 到第741行结束

# 遍历范围内的数据并生成音频
for index in range(start_index, end_index):
    row = df.iloc[index]
    question = row["question"]
    audio_filename = f"audio_{index + 1:03d}.wav"  # 按序号命名音频文件
    audio_filepath = os.path.join(output_audio_dir, audio_filename)

    print(f"Generating audio file: {audio_filename}")
    try:
        # 使用gTTS生成音频
        tts = gTTS(text=question, lang='en')
        tts.save(audio_filepath)
        time.sleep(1)  # 添加延迟，防止请求过多导致错误
    except Exception as e:
        print(f"Error generating audio for row {index + 1}: {e}")
        continue

print(f"Audio files from 1 to {end_index} have been successfully generated and saved to: {output_audio_dir}")