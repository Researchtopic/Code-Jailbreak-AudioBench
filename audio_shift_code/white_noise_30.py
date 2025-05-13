import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

'''
cd /home/ubuntu/Jing/Qwen-Audio/datasets/filtered_MM-Safetybench/audio_original

# 扫描所有 MP3 伪装的 WAV 文件，并转换为真正的 WAV
for file in *.wav; do
    if file "$file" | grep -q "MPEG ADTS"; then
        echo "⚠️ $file is actually an MP3. Converting to true WAV..."
        ffmpeg -i "$file" -acodec pcm_s16le -ar 16000 -y "${file}.fixed.wav"
        mv "${file}.fixed.wav" "$file"  # 覆盖原文件
    fi
done

echo "✅ All MP3 files have been converted to WAV!"
'''

# 定义文件夹路径
original_audio_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/audio_original"
output_folder = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/noise/white_noise"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 设定白噪声音量
white_noise_gain_dB = -30.0

# 遍历所有音频文件
for filename in os.listdir(original_audio_folder):
    if filename.endswith(".wav"):
        original_audio_path = os.path.join(original_audio_folder, filename)

        try:
            # 直接尝试加载原始音频
            original_audio = AudioSegment.from_wav(original_audio_path)
        except:
            # 如果加载失败，则转换格式并覆盖原文件
            print(f"⚠️ Warning: {filename} format issue detected, converting to standard WAV...")

            # 使用 ffmpeg 转换音频格式（覆盖原文件）
            os.system(f"ffmpeg -i \"{original_audio_path}\" -acodec pcm_s16le -ar 44100 -y \"{original_audio_path}\"")

            # 再次尝试加载转换后的音频
            original_audio = AudioSegment.from_wav(original_audio_path)

        # 生成与原始音频相同长度的白噪声
        white_noise = WhiteNoise().to_audio_segment(duration=len(original_audio))

        # 调整白噪声音量
        adjusted_white_noise = white_noise.apply_gain(white_noise_gain_dB)

        # 叠加白噪声
        combined_audio = original_audio.overlay(adjusted_white_noise)

        # 生成新文件名（保持原始音频名，只添加后缀）
        new_filename = f"{os.path.splitext(filename)[0]}_white_noise.wav"
        output_path = os.path.join(output_folder, new_filename)

        # 导出音频
        combined_audio.export(output_path, format="wav")

        print(f"✅ Processed {filename} -> {new_filename}")

print("All files processed with added white noise!")