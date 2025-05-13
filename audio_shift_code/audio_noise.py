import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# 定义文件夹路径
original_audio_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/audio_original"
output_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/noise/white_noise"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有音频文件
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)
        
        # 加载原始音频文件
        original_audio = AudioSegment.from_wav(original_audio_path)
        
        # 获取原始音频的响度（dBFS）
        original_loudness = original_audio.dBFS
        
        # 生成与原始音频长度相同的白噪声
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))
        
        # 调整白噪声的响度与原始音频匹配
        change_in_dBFS = original_loudness - white_noise.dBFS
        adjusted_white_noise = white_noise.apply_gain(change_in_dBFS)
        
        # 将调整后的白噪声与原始音频混合
        combined_audio = original_audio.overlay(adjusted_white_noise)
        
        # 生成新的文件名
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        
        # 导出混合后的音频文件到输出文件夹
        output_path = os.path.join(output_folder, new_filename)
        combined_audio.export(output_path, format="wav")
        
        print(f"Processed {filename} -> {new_filename}")

print("✅ All files processed with added white noise!")