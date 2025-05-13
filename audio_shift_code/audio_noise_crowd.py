import os
from pydub import AudioSegment

# 输入路径
original_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/dataset_grid/accent/black/x2_x0.5_machine"
noise_audio_path = "/home/ubuntu/Jing/Qwen-Audio/noise_original/noise_original_change/machine_noise_normalized_noise.wav"
output_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/dataset_grid/accent/black/x2_x0.5_machine"

# 确保输出目录存在
os.makedirs(output_audio_dir, exist_ok=True)

# 加载噪音音频
noise = AudioSegment.from_file(noise_audio_path)

# 当前噪音音量
current_volume = noise.dBFS

# 目标噪音音量
target_volume = -30.0

# 计算需要的增益
gain = target_volume - current_volume

# 调整噪音音量
adjusted_noise = noise.apply_gain(gain)

# 打印调整信息
print(f"Current Noise Volume: {current_volume:.2f} dBFS")
print(f"Target Noise Volume: {target_volume:.2f} dBFS")
print(f"Applied Gain: {gain:.2f} dB")

# 函数：重复噪音以匹配原始音频长度
def match_noise_length(original, noise):
    if len(noise) >= len(original):
        return noise[:len(original)]  # 噪音足够长，裁剪
    else:
        repeats = len(original) // len(noise) + 1  # 计算需要的重复次数
        extended_noise = noise * repeats  # 重复噪音
        return extended_noise[:len(original)]  # 裁剪到精确长度

# 遍历原始音频文件
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_dir, filename)
        output_audio_path = os.path.join(output_audio_dir, filename.replace(".wav", "_noise_crowd.wav"))
        
        try:
            # 加载原始音频
            original_audio = AudioSegment.from_file(original_audio_path)
            
            # 调整噪音长度与原始音频一致
            noise_to_add = match_noise_length(original_audio, adjusted_noise)
            
            # 混合音频
            mixed_audio = original_audio.overlay(noise_to_add)
            
            # 导出带噪音的音频
            mixed_audio.export(output_audio_path, format="wav")
            print(f"Processed {filename} -> {output_audio_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# 最后打印噪音调整后的音量
print(f"\nFinal Noise Volume Added: {target_volume:.2f} dBFS")