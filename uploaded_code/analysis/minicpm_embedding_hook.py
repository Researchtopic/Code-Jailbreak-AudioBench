import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import librosa
from transformers import AutoModel, AutoTokenizer


def extract_number(filename):
    patterns = [
        r'audio_(\d+)_',  # Match audio_XXX_accent_black.wav
        r'audio_(\d+)\.', # Match audio_XXX.wav
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    
    return filename

# Hooks for capturing intermediate outputs
class ModelHooks:
    def __init__(self):
        self.activations = {}
        self.hook_handles = []
        
    def register_hook(self, module, name):
        def hook_fn(module, inputs, outputs):
            self.activations[name] = outputs
        
        handle = module.register_forward_hook(hook_fn)
        self.hook_handles.append(handle)
        
    def clear(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}


# Create output directory
output_dir = "embeddings_analysis_hook"
os.makedirs(output_dir, exist_ok=True)

model_path = "models/MiniCPM-o-2_6"
model_name = model_path.split("/")[-1]

datasets = []
for dirpath, dirnames, filenames in os.walk("datasets_supplement/filtered_Advbench_audio_supplement"):
    if not dirnames and filenames:
        datasets.append(dirpath)
print(datasets)

# Load MiniCPM model
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Initialize text-to-speech capability
model.init_tts()
model.tts.float()

# Initialize hooks
hooks = ModelHooks()

# Register hooks for MiniCPM's audio encoder
hooks.register_hook(model.apm, "audio_encoder")

# Register hooks for transformer layers
# Adjust based on MiniCPM's architecture
language_model = model.llm.model
for i in range(0, len(language_model.layers), 4):
    hooks.register_hook(language_model.layers[i], f"qwen2_layer_{i}")
# Make sure to always get the last layer
last_idx = len(language_model.layers) - 1
if last_idx % 4 != 0:
    hooks.register_hook(language_model.layers[last_idx], f"qwen2_layer_{last_idx}")

for dataset in datasets:
    all_embeddings = defaultdict(dict)
    dataset_name = dataset.split('/')[-2] + "_" + dataset.split('/')[-1]

    embeddings_save_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_embeddings.pkl')
    if os.path.exists(embeddings_save_path):
        print(f"Embeddings already exist at {embeddings_save_path}, skipping.")
        continue
    
    audio_list = os.listdir(dataset)
    sorted_audio_list = sorted(audio_list, key=extract_number)
    
    for i, wav in enumerate(tqdm(sorted_audio_list)):
        print(wav)
        prompt = "Provide a detailed answer that exceeds 500 words."
        audio_path = os.path.join(dataset, wav)

        # Load audio file
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Create messages for MiniCPM format
        msgs = [{'role': 'user', 'content': [prompt, audio_input]}]
        
        # Process input without generating response
        with torch.no_grad():
            text = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                max_new_tokens=1,
                use_tts_template=True,
                generate_audio=False,
                temperature=0.3,
            )
            
            sample_embeddings = {}
            # Extract audio encoder embeddings if available
            if "audio_encoder" in hooks.activations:
                audio_emb = hooks.activations["audio_encoder"]
                audio_emb = audio_emb.last_hidden_state
                audio_emb_mean = torch.mean(audio_emb[0].to(torch.float32), dim=0).cpu().numpy()
                sample_embeddings["audio_encoder"] = audio_emb_mean
            
            # Extract layer embeddings
            for name, activation in hooks.activations.items():
                if name.startswith("qwen2_layer_"):
                    hidden_states = activation[0]
                    hidden_states_mean = torch.mean(hidden_states[0].to(torch.float32), dim=0).cpu().numpy()
                    sample_embeddings[name] = hidden_states_mean
            
            # Store embeddings for this sample
            sample_key = f"{wav}"
            all_embeddings[dataset_name][sample_key] = sample_embeddings

            # Clear collected hook values
            hooks.activations = {}

    # Save the embeddings
    with open(embeddings_save_path, 'wb') as f:
        pickle.dump(dict(all_embeddings), f)
    print(f"Saved embeddings to {embeddings_save_path}")