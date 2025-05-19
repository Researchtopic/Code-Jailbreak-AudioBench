import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import json
import re
import pickle
from tqdm import tqdm
from pprint import pprint
from itertools import count
from collections import defaultdict

import torch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


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

device = "cuda:0"
model_path = "models/Qwen2-Audio-7B-Instruct"
model_name = model_path.split("/")[-1]

datasets = []
for dirpath, dirnames, filenames in os.walk("datasets_supplement/filtered_Advbench_audio_supplement"):
    if not dirnames and filenames:
        datasets.append(dirpath)
print(datasets)

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map=device)

# Initialize hooks
hooks = ModelHooks()

hooks.register_hook(model.audio_tower, "audio_encoder")

language_model = model.language_model.model
for i in range(0, len(language_model.layers), 4):
    hooks.register_hook(language_model.layers[i], f"qwen2_layer_{i}")
# make sure always get the last layer
last_idx = len(language_model.layers) - 1
if last_idx % 4 != 0:
    hooks.register_hook(language_model.layers[last_idx], f"qwen2_layer_{last_idx}")

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
        audio_path = os.path.join(dataset, wav)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": prompt},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)[0])

        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs.input_ids = inputs.input_ids.to(device)
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=1)
            
            sample_embeddings = {}
            if "audio_encoder" in hooks.activations:
                audio_emb = hooks.activations["audio_encoder"]["last_hidden_state"][0]
                audio_emb_mean = torch.mean(audio_emb, dim=0).cpu().numpy()
                sample_embeddings["audio_encoder"] = audio_emb_mean
            
            for name, activation in hooks.activations.items():
                if name.startswith("qwen2_layer_"):
                    hidden_states = activation[0]   # [1, 184, 4096]
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