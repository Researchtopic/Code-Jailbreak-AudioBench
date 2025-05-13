import os
import librosa
import soundfile as sf

# 设置音频文件夹路径
input_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/audio_original'
output_folder = '/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/speed/x1.5'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有音频文件
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
audio_files.sort()  # 按文件名排序，确保处理顺序一致

# 定义语速倍速
speed_rate = 1.5

# 处理每个音频文件
for audio_file in audio_files:
    input_path = os.path.join(input_folder, audio_file)
    
    # 加载音频文件
    y, sr = librosa.load(input_path)
    
    # 改变语速
    y_changed = librosa.effects.time_stretch(y, rate=speed_rate)
    
    # 提取原始编号（如从 audio_001.wav 提取 001）
    original_id = os.path.splitext(audio_file)[0]  # 去掉扩展名
    new_filename = f"{original_id}_speedx{speed_rate}.wav"  # 按照建议的命名规则生成文件名
    
    # 生成输出文件路径
    output_path = os.path.join(output_folder, new_filename)
    
    # 保存处理后的音频文件
    sf.write(output_path, y_changed, sr)
    
    print(f'Processed {audio_file} -> {new_filename}')

print("✅ All audio files have been processed with new naming conventions.")