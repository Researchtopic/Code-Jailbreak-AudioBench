import os
import subprocess

# 设置输入音频文件夹路径
input_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets/dataset_grid/accent/black/x2_x0.5_machine'

# 设置不同倍速的输出文件夹
output_folders = {
    0.5: '/home/ubuntu/Jing/Qwen-Audio/datasets/dataset_grid/accent/black/x2_x0.5_machine',
    1.5: '/home/ubuntu/Jing/Qwen-Audio/datasets/dataset_grid/accent/black/x2_x1.5_machine'
}

# 确保输出文件夹存在
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# 获取所有 .wav 音频文件
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # 按文件名排序，确保处理顺序一致

# 需要改变的语速
speed_rates = [0.5, 1.5]

# 处理每个音频文件
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)

    for speed_rate in speed_rates:
        # 提取原始编号（如 audio_001.wav 提取 001）
        original_id = os.path.splitext(audio_file)[0]
        new_filename = f"{original_id}_speedx{speed_rate}.wav"

        # 生成输出文件路径
        output_path = os.path.join(output_folders[speed_rate], new_filename)

        # 使用 sox 进行变速但保持音高
        cmd = ["sox", input_path, output_path, "tempo", "-s", str(speed_rate)]
        subprocess.run(cmd, check=True)

        print(f'Processed {audio_file} -> {new_filename} (Speed x{speed_rate})')

print(" All audio files have been processed using SOX (tempo).")