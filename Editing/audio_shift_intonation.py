import os
import librosa
import soundfile as sf
import numpy as np

# Input and output paths
original_audio_dir = "filtered_datasets/filtered_advbench/audio_original"
output_audio_dir = "filtered_datasets/filtered_advbench/intonation/+4"
os.makedirs(output_audio_dir, exist_ok=True)

# Define pitch changes for current experiment (semitone level)
semitone_shifts = [0, 4, 8, 12]

# Function: apply intonation changes to audio by uniform segments
def apply_intonation_changes(input_path, output_path, semitone_shifts):
    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    
    # Get total audio duration
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Calculate duration of each segment
    segment_duration = total_duration / len(semitone_shifts)
    
    # Initialize result array
    processed_audio = []

    for i, semitone in enumerate(semitone_shifts):
        # Calculate start and end sample points for current segment
        start_sample = int(i * segment_duration * sr)
        end_sample = int((i + 1) * segment_duration * sr) if i < len(semitone_shifts) - 1 else len(y)
        
        # Extract current segment
        segment = y[start_sample:end_sample]
        
        # Adjust pitch (keep speed unchanged)
        shifted_segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitone)
        
        # Add to result array
        processed_audio.append(shifted_segment)
    
    # Combine all processed audio segments
    output_audio = np.concatenate(processed_audio)
    
    # Save processed audio
    sf.write(output_path, output_audio, sr)

# Traverse original audio files
for filename in os.listdir(original_audio_dir):
    if filename.endswith(".wav"):
        input_audio_path = os.path.join(original_audio_dir, filename)

        # Set output filename
        output_filename = filename.replace(".wav", "_intonation_04812.wav")
        output_audio_path = os.path.join(output_audio_dir, output_filename)

        try:
            # Apply pitch changes
            apply_intonation_changes(input_audio_path, output_audio_path, semitone_shifts)
            print(f"Processed {filename} -> {output_audio_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")