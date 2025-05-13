from pydub import AudioSegment
import os
from pathlib import Path

base_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/filtered_datasets/filtered_Advbench"
output_base_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets/datasets_with_defense/filtered_Advbench_audio_with_defense"
defense_prompt_path = "/home/ubuntu/Jing/Qwen-Audio/datasets/defense_prompt/defense_prompt.wav"

#添加defense audio
defense_audio = AudioSegment.from_wav(defense_prompt_path)
silence = AudioSegment.silent(duration=1000)  # 1秒静音

#遍历
for dirpath, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if not filename.endswith(".wav"):
            continue

        full_input_path = os.path.join(dirpath, filename)
        relative_path = os.path.relpath(full_input_path, base_dir)
        full_output_path = os.path.join(output_base_dir, relative_path)

        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)

        try:
            original_audio = AudioSegment.from_wav(full_input_path)
            combined_audio = defense_audio + silence + original_audio

            #时长
            len_defense = len(defense_audio)
            len_silence = len(silence)
            len_original = len(original_audio)
            len_combined = len(combined_audio)
            expected_total = len_defense + len_silence + len_original

            print(f"{relative_path}")
            print(f" 防御音频: {len_defense} ms")
            print(f" 静音: {len_silence} ms")
            print(f" 原始音频: {len_original} ms")
            print(f" 预期合计: {expected_total} ms")
            print(f" 实际合计: {len_combined} ms")
            print("-" * 50)

            combined_audio.export(full_output_path, format="wav")

        except Exception as e:
            print(f"拼接失败: {relative_path}，原因: {e}")