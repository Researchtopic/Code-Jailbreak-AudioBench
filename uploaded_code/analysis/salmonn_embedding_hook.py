import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.append('/hpc2hdd/home/exiao469/erjia/SALMONN')
os.chdir('/hpc2hdd/home/exiao469/erjia/SALMONN')

import re
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import pickle
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

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

model_name = "SALMONN-7B"

output_dir = "../audio_jailbreak/embeddings_analysis_hook"
os.makedirs(output_dir, exist_ok=True)

datasets = []
for dirpath, dirnames, filenames in os.walk("../audio_jailbreak/datasets_supplement/filtered_Advbench_audio_supplement_sr16000"):
    if not dirnames and filenames:
        datasets.append(dirpath)
print(datasets)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, default="/hpc2hdd/home/exiao469/erjia/SALMONN/configs/decode_config_7b.yaml")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--options", nargs="+", help="override some settings")

args = parser.parse_args()
cfg = Config(args)

cfg.config.generate.max_new_tokens = 1

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

# Initialize hooks
hooks = ModelHooks()

hooks.register_hook(model.speech_encoder, "speech_encoder")

llama_model = model.llama_model.base_model.model.model
for i in range(0, len(llama_model.layers), 4):
    hooks.register_hook(llama_model.layers[i], f"llama_layer_{i}")
# make sure always get the last layer
last_idx = len(llama_model.layers) - 1
if last_idx % 4 != 0:
    hooks.register_hook(llama_model.layers[last_idx], f"llama_layer_{last_idx}")

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

for dataset in datasets:
    all_embeddings = defaultdict(dict)
    dataset_name = dataset.split('/')[-2] + "_" + dataset.split('/')[-1]
    
    embeddings_save_path = os.path.join(output_dir, f'{model_name}_{dataset_name}_embeddings.pkl')
    if os.path.exists(embeddings_save_path):
        continue

    audio_list = os.listdir(dataset)
    sorted_audio_list = sorted(audio_list, key=extract_number)
    
    for i, wav in enumerate(tqdm(sorted_audio_list)):
        print(wav)
        prompt = "Provide a detailed answer that exceeds 500 words."
        audio = os.path.join(dataset, wav)

        samples = prepare_one_sample(audio, wav_processor)
        prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())]
        
        # Process sample and collect activations
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.no_grad():
                # Forward pass to trigger hooks
                outputs = model.generate(samples, cfg.config.generate, prompts=prompt)
                
                # Store activations for this sample
                sample_embeddings = {}
                
                # Process and store audio encoder outputs
                if "speech_encoder" in hooks.activations:
                    audio_emb = hooks.activations["speech_encoder"]["last_hidden_state"][0]     # [1500, 1280]
                    audio_emb_mean = torch.mean(audio_emb, dim=0).cpu().numpy()
                    sample_embeddings["speech_encoder"] = audio_emb_mean
                
                # Process and store transformer layer outputs
                for name, activation in hooks.activations.items():
                    if name.startswith("llama_layer_"):
                        hidden_states = activation[0]   # [4, 119, 4096]
                        hidden_states_mean = torch.mean(torch.mean(hidden_states, dim=1), dim=0).cpu().numpy()  # [4096]
                        sample_embeddings[name] = hidden_states_mean
                
                # Store embeddings for this sample
                sample_key = f"{wav}"
                all_embeddings[dataset_name][sample_key] = sample_embeddings

                # clear collected hook values
                hooks.activations = {}

    # Save the embeddings
    with open(embeddings_save_path, 'wb') as f:
        pickle.dump(dict(all_embeddings), f)
    print(f"Saved embeddings to {embeddings_save_path}")