import os
from pydub import AudioSegment

# 输入路径
original_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_machine"
noise_audio_path = "/home/ubuntu/Jing/Qwen-Audio/noise_original/noise_original_change/machine_noise_normalized_noise.wav"
output_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_machine"

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

print(f"Current Noise Volume: {current_volume:.2f} dBFS")
print(f"Target Noise Volume: {target_volume:.2f} dBFS")
print(f"Applied Gain: {gain:.2f} dB")

# 函数：重复噪音以匹配原始音频长度
def match_noise_length(original, noise):
    if len(noise) >= len(original):
        return noise[:len(original)]
    else:
        repeats = len(original) // len(noise) + 1
        extended_noise = noise * repeats
        return extended_noise[:len(original)]

# 遍历原始音频文件
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav") and "_machine" not in filename:
        original_audio_path = os.path.join(original_audio_dir, filename)
        output_audio_path = os.path.join(output_audio_dir, filename.replace(".wav", "_machine.wav"))
        
        try:
            original_audio = AudioSegment.from_file(original_audio_path)
            noise_to_add = match_noise_length(original_audio, adjusted_noise)
            mixed_audio = original_audio.overlay(noise_to_add)
            mixed_audio.export(output_audio_path, format="wav")

            # 删除原始音频文件
            os.remove(original_audio_path)

            print(f"Processed and deleted: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"\nFinal Noise Volume Added: {target_volume:.2f} dBFS")