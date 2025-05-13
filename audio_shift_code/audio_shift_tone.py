import os
import librosa
import soundfile as sf
from tqdm import tqdm  # 用于显示进度条

# 输入和输出文件夹路径
input_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/audio_original_2'  # 原始音频文件夹
output_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/tone/test_2'  # 目标音调的输出文件夹

# 如果输出文件夹不存在，创建文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义变调的半音数量
semitone_variation = +4 # 正半音变化

# 获取输入文件夹中的所有 .wav 文件
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # 确保按照文件名顺序处理

print(f"📂 发现 {len(audio_files)} 个音频文件，正在生成 tone 变体...")

# 遍历每个音频文件并生成变调文件
for filename in tqdm(audio_files, desc="Processing..."):
    try:
        # 加载音频文件
        input_path = os.path.join(input_folder, filename)
        y, sr = librosa.load(input_path, sr=None)  # 保持原始采样率

        # 变调
        y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=semitone_variation)

        # 生成新的文件名：audio_<编号>_tone+12st.wav
        original_id = os.path.splitext(filename)[0]  # 提取原始编号（如 001）
        new_filename = f"{original_id}_tone+4st.wav"
        output_path = os.path.join(output_folder, new_filename)

        # 保存变调后的音频
        sf.write(output_path, y_shifted, sr)
    except Exception as e:
        print(f"❌ 处理 {filename} 失败，错误：{e}")

print("✅ 所有音频文件已生成完毕！")