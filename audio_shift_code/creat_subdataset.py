import pandas as pd
import numpy as np
import math
import os
import random
import shutil

random.seed(42)

datasets_config = {
    "Advbench": {
        "target_count": 58,
        "id_column": "id",
        "category_column": "category"
    },
    "MM-Safetybench": {
        "target_count": 77,
        "id_column": "id", 
        "category_column": "category"
    },
    "RedTeam_2K": {
        "target_count": 97,
        "id_column": "id",
        "category_column": "from"
    },
    "safebench": {
        "target_count": 30,
        "id_column": "task_id",
        "category_column": "category_name"
    }
}

base_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/strict_part_screams"  
csv_dir = "/home/ubuntu/Jing/Qwen-Audio/Datasets_Text/filtered"  
output_dir = "/home/ubuntu/Jing/Qwen-Audio/datasets_supplement/strict_small_screams"  
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)

for dataset_name, config in datasets_config.items():
    print(f"\nProcessing dataset: {dataset_name}")
    
    csv_path = os.path.join(csv_dir, f"filtered_{dataset_name}.csv")
    print(f"Reading CSV file: {csv_path}")
    
    try:
        strict_df = pd.read_csv(csv_path)
        print(f"CSV columns: {strict_df.columns.tolist()}")
        print(f"CSV rows: {len(strict_df)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        continue
    
    target_count = config["target_count"]
    id_column = config["id_column"]
    category_column = config["category_column"]
    
    missing_columns = []
    if id_column not in strict_df.columns:
        missing_columns.append(id_column)
    if category_column not in strict_df.columns:
        missing_columns.append(category_column)
    if "question" not in strict_df.columns:
        missing_columns.append("question")
    
    if missing_columns:
        print(f"Warning: Missing required columns: {', '.join(missing_columns)}")
        continue
    
    selected_rows = []
    
    category_counts = strict_df[category_column].value_counts()
    print(f"Category distribution (using {category_column} column):")
    for cat, count in category_counts.items():
        select_count = math.ceil(count / 10)
        print(f"  - {cat}: Total {count}, Selecting {select_count}")
    
    for category, group in strict_df.groupby(category_column):
        select_count = math.ceil(len(group) / 10)
        select_count = min(select_count, len(group))
        try:
            selected = group.sample(n=select_count, random_state=42)
            selected_rows.append(selected)
        except Exception as e:
            print(f"Error selecting samples for category {category}: {e}")
    
    selected_df = pd.concat(selected_rows) if selected_rows else pd.DataFrame()
    
    actual_count = len(selected_df)
    print(f"Selected {actual_count} samples from {dataset_name} (Target: {target_count})")
    
    if actual_count != target_count:
        print(f"Warning: Selected count ({actual_count}) doesn't match target count ({target_count})")
    
   
    source_audio_dir = os.path.join(base_dir, f"filtered_{dataset_name}", "screams")

    print(f"Source audio directory: {source_audio_dir}")
    
    if not os.path.exists(source_audio_dir):
        print(f"Error: Source audio directory doesn't exist: {source_audio_dir}")
        continue
    
    copied_count = 0
    error_count = 0
    
    for _, row in selected_df.iterrows():
        file_id = str(row[id_column])
        
        possible_file_paths = [
    os.path.join(source_audio_dir, f"audio_{file_id.zfill(3)}_emotion_screams.wav"),
    os.path.join(source_audio_dir, f"audio_{file_id}_emotion_screams.wav"),
]

        
        found_file = False
        for source_audio in possible_file_paths:
            if os.path.exists(source_audio):
                audio_filename = os.path.basename(source_audio)
                target_audio = os.path.join(output_dir, "audio", f"{dataset_name}_{audio_filename}")
                
                try:
                    shutil.copy2(source_audio, target_audio)
                    copied_count += 1
                    if copied_count <= 5:
                        print(f"Copied: {os.path.basename(source_audio)} → {os.path.basename(target_audio)}")
                    found_file = True
                    break
                except Exception as e:
                    print(f"Error copying file: {e}")
                    error_count += 1
        
        if not found_file:
            error_count += 1
            if error_count <= 5:
                print(f"Warning: Couldn't find audio file for ID {file_id}")
                try:
                    some_files = os.listdir(source_audio_dir)[:5]
                    print(f"  Some files in source directory: {some_files}")
                except Exception:
                    pass
    
    print(f"Successfully copied {copied_count} files, failed {error_count}")
    
    output_csv = os.path.join(output_dir, f"{dataset_name}_strict_small.csv")
    selected_df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")
    print("-" * 50)

print("Strict-small datasets was created!")