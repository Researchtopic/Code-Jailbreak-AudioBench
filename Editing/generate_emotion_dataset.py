import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('/path/to/dia_project')
os.chdir('/path/to/dia_project')

import csv
import torch
import random
import shutil
import numpy as np
from dia.model import Dia


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN (if used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1024)

# Define CSV files and their corresponding question column indices
csv_files = [
    "/path/to/audio_jailbreak_data/datasets_text/unselected/unselected_MM-Safetybench.csv",
    "/path/to/audio_jailbreak_data/datasets_text/unselected/unselected_RedTeam_2K.csv",
    "/path/to/audio_jailbreak_data/datasets_text/unselected/unselected_safebench.csv",
]

# Column indices for questions in each CSV (0-indexed)
question_column_indices = [1, 1, 4]

# Define suffixes for audio variations
# suffixes = ["laughs", "screams", "groans", "singing", "sings", "humming", "whistles", "beep", "applause"]
# suffixes = ["coughs", "clears throat", "laughs", "screams"]
suffixes = ["laughs", "screams"]

# Load model
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=torch.device("cuda"))

# Process each CSV file
for csv_idx, csv_path in enumerate(csv_files):
    # Extract the CSV filename without extension to use as directory name
    csv_filename = os.path.basename(csv_path)
    csv_name = os.path.splitext(csv_filename)[0]
    
    # Create main directory for this CSV
    csv_dir = os.path.join(os.getcwd(), csv_name)
    if os.path.exists(csv_dir):
        # print(f"Removing existing directory: {csv_dir}")
        # shutil.rmtree(csv_dir)
        print(f"directory exists : {csv_dir}")
        sys.exit()
    os.makedirs(csv_dir, exist_ok=True)
    
    # Create emotion directory
    emotion_dir = os.path.join(csv_dir, "emotion")
    os.makedirs(emotion_dir, exist_ok=True)
    
    # Create subdirectories for different suffixes under emotion directory
    for suffix in suffixes:
        suffix_dir = os.path.join(emotion_dir, suffix)
        os.makedirs(suffix_dir, exist_ok=True)
    
    # Read questions from current CSV file
    texts = []
    question_col_idx = question_column_indices[csv_idx]
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # Skip header row
        next(csv_reader)
        # Get questions from the specified column
        for row in csv_reader:
            if row and len(row) > question_col_idx:  # Check if row has enough columns
                texts.append(row[question_col_idx])
    
    print(f"Successfully loaded {len(texts)} questions from {csv_path}")
    
    # Generate audio for each question with each suffix
    for suffix in suffixes:
        for i, text in enumerate(texts, 1):
            # Format the file name with leading zeros
            file_name = f"audio_{i:03d}.wav"
            file_path = os.path.join(emotion_dir, suffix, file_name)
            
            # Add suffix to the text
            text_with_suffix = "[S1] " + text + f" ({suffix}) [S2]"
            
            clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face. "
            clone_from_audio = "simple.mp3"

            print(f"Generating audio for: {text_with_suffix}")
            # output_audio_np = model.generate(text_with_suffix, use_torch_compile=True, verbose=True)
            output_audio_np = model.generate(clone_from_text + text_with_suffix, audio_prompt=clone_from_audio, use_torch_compile=True, verbose=True)

            # Get sample rate from the loaded DAC model
            output_sr = 44100
            
            # --- Slow down audio ---
            speed_factor = 0.9
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor
            
            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio_np = resampled_audio_np.astype(np.float32)
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")

            # Save audio file
            print(f"Saving audio to: {file_path}")
            model.save_audio(file_path, output_audio_np)

            # if i >= 10:
            #     break

print("All audio files generated successfully!")