import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# 输入原始音频文件夹
original_audio_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise"
# 输出带白噪声的音频文件夹
output_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/dataset_grid/emotion/screams/x5_x1.5_white_noise"

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 设置白噪声目标音量（单位 dBFS）
white_noise_gain_dB = -30.0

# 遍历所有 .wav 文件
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)

        try:
            original_audio = AudioSegment.from_wav(original_audio_path)
        except:
            print(f"⚠️ Warning: {filename} format issue detected, converting to standard WAV...")
            os.system(f"ffmpeg -i \"{original_audio_path}\" -acodec pcm_s16le -ar 44100 -y \"{original_audio_path}\"")
            original_audio = AudioSegment.from_wav(original_audio_path)

        # 生成与原始音频等长的白噪声
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))
        adjusted_white_noise = white_noise.apply_gain(white_noise_gain_dB)

        # 叠加白噪声
        combined_audio = original_audio.overlay(adjusted_white_noise)

        # 命名新文件
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        output_path = os.path.join(output_folder, new_filename)

        # 导出音频
        combined_audio.export(output_path, format="wav")

        # 删除原始音频文件
        os.remove(original_audio_path)

        print(f"✅ Processed {filename} -> {new_filename} and deleted original.")

print("All files processed with added white noise and originals deleted.")