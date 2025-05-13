import os
import librosa
import soundfile as sf

# 输入和输出文件夹路径
input_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/strict_small_screams/audio'  # 输入文件夹路径
output_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise'  # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有音频文件
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        filepath = os.path.join(input_folder, filename)
        
        # 加载音频文件
        y, sr = librosa.load(filepath)
        
        # 定义加重音的时间段（开头1秒）
        start_time = 0.0
        end_time = 1.0
        
        # 将时间转换为样本点
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # 增加音量
        gain = 5.0  # 倍数
        y[start_sample:end_sample] *= gain
        
        # 生成新的文件名
        base_name, ext = os.path.splitext(filename)
        new_filename = f"{base_name}_emphasis_x5{ext}"
        
        # 保存处理后的音频
        output_path = os.path.join(output_folder, new_filename)
        sf.write(output_path, y, sr)
        print(f"Processed and saved: {output_path}")


print("All audio files have been processed.")