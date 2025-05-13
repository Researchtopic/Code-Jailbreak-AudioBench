import os
import librosa
import soundfile as sf
import numpy as np

# 输入路径和输出路径
original_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/audio_original"
output_audio_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_advbench/intonation/+4"
os.makedirs(output_audio_dir, exist_ok=True)

# 定义当前实验的音高变化（半音级别）
semitone_shifts = [0, 4, 8, 12]

# 函数：对音频按均匀分段调整音高
def apply_intonation_changes(input_path, output_path, semitone_shifts):
    # 加载音频
    y, sr = librosa.load(input_path, sr=None)
    
    # 获取音频总时长
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # 计算每段的持续时间
    segment_duration = total_duration / len(semitone_shifts)
    
    # 初始化结果数组
    processed_audio = []

    for i, semitone in enumerate(semitone_shifts):
        # 计算当前段的起始和结束采样点
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr) if i < len(semitone_shifts) - 1 else len(y)
        
        # 提取当前段
        segment = y[start_sample:end_sample]
        
        # 调整音高（保持速度不变）
        shifted_segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitone)
        
        # 添加到结果数组
        processed_audio.append(shifted_segment)
    
    # 合并所有处理后的音频段
    output_audio = np.concatenate(processed_audio)
    
    # 保存处理后的音频
    sf.write(output_path, output_audio, sr)

# 遍历原始音频文件
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav"):
        input_audio_path = os.path.join(original_audio_dir, filename)

        # 设置输出文件名
        output_filename = filename.replace(".wav", "_intonation_04812.wav")
        output_audio_path = os.path.join(output_audio_dir, output_filename)

        try:
            # 应用音高变化
            apply_intonation_changes(input_audio_path, output_audio_path, semitone_shifts)
            print(f"Processed {filename} -> {output_audio_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")