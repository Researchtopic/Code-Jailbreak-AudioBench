import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from umap import UMAP
import glob


def extract_layer_embeddings(embeddings_dict, target_layer):
    layer_embeddings = {}
    
    for category, audio_dict in embeddings_dict.items():
        current_embeddings = []
        for audio_id, layers_dict in audio_dict.items():
            # Find all layer keys that contain the target_layer keyword
            matching_layers = [layer_key for layer_key in layers_dict.keys() if target_layer in layer_key]
            
            # If there are multiple matches, we'll use the first one
            if matching_layers:
                # Use the first matching layer
                matched_layer = matching_layers[0]
                current_embeddings.append(layers_dict[matched_layer])
        
        if current_embeddings:
            layer_embeddings[category] = current_embeddings
            
    return layer_embeddings


def visualize_embeddings(embeddings_dict, method, model_name, output_dir, target_types, target_layer):
    # Extract embeddings for the specified layer
    layer_embeddings = extract_layer_embeddings(embeddings_dict, target_layer)
    
    modified_dict = {}
    for key, value in layer_embeddings.items():
        if "original" in key.lower():
            modified_dict["original"] = value
        else:
            modified_dict[key] = value
    layer_embeddings = modified_dict

    if len(target_types) > 2:
        combined_embeddings = {}
        for category in target_types:
            combined_embeddings[category] = []
            
        for key, embeddings in layer_embeddings.items():
            if key == 'original':
                combined_embeddings['original'] = embeddings
                continue
                
            for category in target_types:
                if category != 'original' and key.startswith(category):
                    combined_embeddings[category].extend(embeddings)
                    break

        layer_embeddings = combined_embeddings

    if len(target_types) > 2:
        group_types = [key for key in layer_embeddings.keys()]
    else:
        group_types = sorted(key for key in layer_embeddings.keys())

    if "original" in group_types:
        group_types.remove("original")
        group_types.append("original")

    sns.set_theme(style="ticks")
    plt.figure(figsize=(12, 10))
    
    all_embeddings = []
    group_labels = []

    group_embeddings = {}
    
    for group_type in group_types:
        embeddings = layer_embeddings[group_type]
        all_embeddings.extend(embeddings)
        group_labels.extend([group_type] * len(embeddings))
        group_embeddings[group_type] = embeddings
        
    all_embeddings = np.array(all_embeddings)
    
    if model_name == "SALMONN-7B":
        model_name = "SALMONN-7B"
    elif model_name == "Qwen2-Audio-7B-Instruct":
        model_name = "Qwen2-Audio-7B"
    elif model_name == "MiniCPM-o-2_6":
        model_name = "MiniCPM-o-2.6"
    elif model_name == "SpeechGPT":
        model_name = "SpeechGPT"

    if "encoder" in target_layer:
        target_layer = "Encoder"
    elif "layer" in target_layer:
        digit_match = re.search(r'layer_(\d+)', target_layer)
        digit = digit_match.group(1)
        target_layer = "Layer " + digit

    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=1024)
        title = f'{model_name}  {target_layer}'
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=2, random_state=1024)
        title = f'UMAP Visualization ({model_name} - {target_layer})'
    else:
        reducer = PCA(n_components=2, random_state=1024)
        title = f'PCA Visualization ({model_name} - {target_layer})'
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    colors = sns.color_palette('husl', n_colors=len(group_types))
    
    legend_handles = []
    
    plt.subplots_adjust(bottom=0.2)
    
    reduced_points_by_group = {}
    start_idx = 0
    
    for i, group_type in enumerate(group_types):
        if group_type not in group_embeddings:
            continue
            
        group_size = len(group_embeddings[group_type])
        end_idx = start_idx + group_size
        
        reduced_points_by_group[group_type] = reduced_embeddings[start_idx:end_idx]
        
        mask = np.array(group_labels) == group_type
        scatter = plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c=[colors[i]], label=group_type.title(), alpha=0.9, s=80, edgecolors='white', linewidths=0.9,)
        
        legend_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], alpha=0.9, markersize=12, label=group_type.title())
        legend_handles.append(legend_handle)
        
        start_idx = end_idx
    
    plt.title(title, fontsize=50, pad=10)
    plt.xlabel('')
    plt.ylabel('')
    
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

    if "Encoder" in target_layer:
        target_layer = "Audio Encoder"

    save_path = os.path.join(output_dir, f"{model_name}_{method}_{target_layer}_{','.join(target_types)}_visualization.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"{save_path} processed.")


if __name__ == "__main__":
    model_names = ["SALMONN-7B", "Qwen2-Audio-7B-Instruct", "MiniCPM-o-2_6"]

    data_dir = "embeddings_analysis_hook"

    output_dir = os.path.join(data_dir, "single_plot")
    os.makedirs(output_dir, exist_ok=True)

    target_types_list = [["emphasis", "speed", "intonation", "tone", "noise", "accent", "emotion"]]

    target_layers = ["encoder", "layer_0", "layer_8", "layer_16", "layer_24", "layer_31"]
    target_layers_minicpm = ["encoder", "layer_0", "layer_8", "layer_16", "layer_24", "layer_27"]

    methods = ["tsne"]

    for target_types in target_types_list:
        target_types.append("original")

        for method in methods:
            for model_name in model_names:
                current_target_layers = []
                if model_name == "MiniCPM-o-2_6":
                    current_target_layers = target_layers_minicpm
                else:
                    current_target_layers = target_layers

                for target_layer in current_target_layers:
                    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
                    
                    matched_files = []
                    for target_type in target_types:
                        for pkl_file in pkl_files:
                            basename = os.path.basename(pkl_file)
                            if model_name in basename and target_type in basename:
                                matched_files.append(pkl_file)
                    
                    merged_embeddings = {}
                    for pkl_file in matched_files:
                        with open(pkl_file, 'rb') as f:
                            embeddings_dict = pickle.load(f)
                        for key, value in embeddings_dict.items():
                            if key not in merged_embeddings:
                                merged_embeddings[key] = {}
                            merged_embeddings[key].update(value)
                    
                    visualize_embeddings(merged_embeddings, method, model_name, output_dir, target_types, target_layer)