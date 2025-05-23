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
    
    # Read audio
    waveform, sample_rate = torchaudio.load(input_path)
    
    # If current sample rate is already the target sample rate, copy file directly
    if sample_rate == new_sample_rate:
        shutil.copy2(input_path, output_path)
        return
    
    # Move audio data to specified device
    waveform = waveform.to(device)
    
    # Create resampler and move to specified device
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    ).to(device)
    
    # Resample
    resampled_waveform = resampler(waveform)
    
    # Move data back to CPU for saving
    resampled_waveform = resampled_waveform.cpu()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save audio
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
    
    # Ensure output root directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Recursively traverse all files and folders
    file_list = list(input_dir.rglob("*"))
    for item in tqdm(file_list):
        # Calculate relative path to maintain same structure in output directory
        relative_path = item.relative_to(input_dir)
        output_path = output_dir / relative_path
        
        # If it's a directory, create corresponding output directory
        if item.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            continue
            
        # Process files
        if item.suffix.lower() == '.wav':
            # Resample WAV files
            resample_audio(str(item), str(output_path), new_sample_rate, device)
            # print(f"Resampled: {relative_path}")
        else:
            # Copy non-WAV files directly
            shutil.copy2(str(item), str(output_path))
            # print(f"Copied: {relative_path}")


if __name__ == "__main__":
    sample_rate = 16000
    input_directory = "datasets/filtered_RedTeam_2K_audio"
    output_directory = f"datasets/filtered_RedTeam_2K_audio_sr{sample_rate}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process_directory(input_directory, output_directory, sample_rate, device)