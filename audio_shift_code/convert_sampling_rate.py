import os
from pathlib import Path
import shutil
import torchaudio
import torch
import time
from tqdm import tqdm


def resample_audio(input_path, output_path, new_sample_rate, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 读取音频
    waveform, sample_rate = torchaudio.load(input_path)
    
    # 如果当前采样率已经是目标采样率，直接复制文件
    if sample_rate == new_sample_rate:
        shutil.copy2(input_path, output_path)
        return
    
    # 将音频数据移动到指定设备
    waveform = waveform.to(device)
    
    # 创建重采样器并移动到指定设备
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    ).to(device)
    
    # 重采样
    resampled_waveform = resampler(waveform)
    
    # 将数据移回CPU用于保存
    resampled_waveform = resampled_waveform.cpu()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存音频
    torchaudio.save(
        output_path,
        resampled_waveform,
        new_sample_rate
    )

def process_directory(input_dir, output_dir, new_sample_rate, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 确保输出根目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 递归遍历所有文件和文件夹
    file_list = list(input_dir.rglob("*"))
    for item in tqdm(file_list):
        # 计算相对路径，用于在输出目录中保持相同的结构
        relative_path = item.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        # 如果是目录，创建对应的输出目录
        if item.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            continue
            
        # 处理文件
        if item.suffix.lower() == '.wav':
            # WAV文件进行重采样
            resample_audio(str(item), str(output_path), new_sample_rate, device)
            # print(f"Resampled: {relative_path}")
        else:
            # 非WAV文件直接复制
            shutil.copy2(str(item), str(output_path))
            # print(f"Copied: {relative_path}")


if __name__ == "__main__":
    sample_rate = 16000
    input_directory = "/hpc2hdd/home/exiao469/erjia/audio_jailbreak/datasets/filtered_RedTeam_2K_audio"
    output_directory = f"/hpc2hdd/home/exiao469/erjia/audio_jailbreak/datasets/filtered_RedTeam_2K_audio_sr{sample_rate}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process_directory(input_directory, output_directory, sample_rate, device)