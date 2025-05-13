import os
import subprocess
from tqdm import tqdm  # 用于显示进度条

# 输入音频文件夹路径
input_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/audio_original"

# 定义输出文件夹，按变调级别存放
output_base = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/tone"
tone_variations = {
    "+4st": os.path.join(output_base, "+4"),
    "-4st": os.path.join(output_base, "-4"),
    "+8st": os.path.join(output_base, "+8"),
    "-8st": os.path.join(output_base, "-8"),
}

# 确保所有输出文件夹存在
for folder in tone_variations.values():
    os.makedirs(folder, exist_ok=True)

# 获取所有 .wav 文件
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()

print(f"📂 发现 {len(audio_files)} 个音频文件，正在使用 SOX 进行整体变调...")

# 遍历每个音频文件，并生成所有变调版本
for i, filename in tqdm(enumerate(audio_files, start=1), desc="Processing..."):
    try:
        input_path = os.path.join(input_folder, filename)
        original_id = f"audio_{i:03d}"  # 修改编号为 3 位格式，如 audio_001

        for tone, output_folder in tone_variations.items():
            # 计算 sox 变调的 pitch 值
            semitone = int(tone.replace("st", "").replace("+", ""))  # 提取数值
            pitch_shift = semitone * 100  # Sox 以 cents（1/100 半音）为单位

            # 生成输出文件路径
            new_filename = f"{original_id}_tone{tone}.wav"
            output_path = os.path.join(output_folder, new_filename)

            # 调用 sox 进行变调
            cmd = ["sox", input_path, output_path, "pitch", str(pitch_shift)]
            subprocess.run(cmd, check=True)

        print(f"✅ 处理完成 {filename}")

    except Exception as e:
        print(f"❌ 处理 {filename} 失败，错误：{e}")

print("🎉 ✅ 所有音频文件已整体变调并存入对应文件夹！")