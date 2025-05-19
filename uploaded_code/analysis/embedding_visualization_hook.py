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
    
    # 修改original的键值，便于可视化
    modified_dict = {}
    for key, value in layer_embeddings.items():
        if "original" in key.lower():
            modified_dict["original"] = value
        else:
            modified_dict[key] = value
    layer_embeddings = modified_dict

    # 如果绘制的是所有类别，则合并同类别下的不同settings
    if len(target_types) > 2:
        combined_embeddings = {}
        for category in target_types:
            # 对于每个目标类别，初始化空列表
            combined_embeddings[category] = []
            
        # 将layer_embeddings中的所有键值对归类到相应的类别中
        for key, embeddings in layer_embeddings.items():
            # 对于'original'类别直接添加
            if key == 'original':
                combined_embeddings['original'] = embeddings
                continue
                
            # 对于其他类别，查找前缀匹配
            for category in target_types:
                if category != 'original' and key.startswith(category):
                    combined_embeddings[category].extend(embeddings)
                    break

        layer_embeddings = combined_embeddings

    if len(target_types) > 2:
        group_types = [key for key in layer_embeddings.keys()]
    else:
        # 将每种editing内的具体setting排序
        group_types = sorted(key for key in layer_embeddings.keys())

    # 把 original 排在列表最后，可视化时绘制的点不会被覆盖
    if "original" in group_types:
        group_types.remove("original")
        # group_types.insert(0, "original")
        group_types.append("original")

    sns.set_theme(style="ticks")
    plt.figure(figsize=(12, 10))
    
    all_embeddings = []
    group_labels = []

    # 创建一个字典来存储每个组的嵌入向量
    group_embeddings = {}
    
    # 为每个组类型准备数据
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

    # 降维
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=1024)
        # title = f't-SNE Visualization ({model_name} - {target_layer})'
        title = f'{model_name}  {target_layer}'
    elif method.lower() == 'umap':
        reducer = UMAP(n_components=2, random_state=1024)
        title = f'UMAP Visualization ({model_name} - {target_layer})'
    else:
        reducer = PCA(n_components=2, random_state=1024)
        title = f'PCA Visualization ({model_name} - {target_layer})'
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # 设置颜色方案
    colors = sns.color_palette('husl', n_colors=len(group_types))
    
    # 创建空的散点图，只用于图例
    legend_handles = []
    
    # 创建子图，并设置底部边距
    plt.subplots_adjust(bottom=0.2)  # 为图例预留足够的底部空间
    
    # 创建一个字典来存储每个组的降维后的点
    reduced_points_by_group = {}
    start_idx = 0
    
    # 绘制散点图并存储每个组的降维后的点
    for i, group_type in enumerate(group_types):
        if group_type not in group_embeddings:
            continue
            
        group_size = len(group_embeddings[group_type])
        end_idx = start_idx + group_size
        
        # 存储该组的降维后的点
        reduced_points_by_group[group_type] = reduced_embeddings[start_idx:end_idx]
        
        # 绘制散点图
        mask = np.array(group_labels) == group_type
        scatter = plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], c=[colors[i]], label=group_type.title(), alpha=0.9, s=80, edgecolors='white', linewidths=0.9,)
        
        # 创建一个更大点的散点对象，仅用于图例，并设置相同的透明度
        legend_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], alpha=0.9, markersize=12, label=group_type.title())
        legend_handles.append(legend_handle)
        
        start_idx = end_idx
    
    plt.title(title, fontsize=50, pad=10)
    # plt.xlabel('Component 1', fontsize=25)
    # plt.ylabel('Component 2', fontsize=25)
    plt.xlabel('')
    plt.ylabel('')
    
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    plt.xticks([])
    plt.yticks([])
    
    # 加粗图表边框
    ax = plt.gca()  # 获取当前轴
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)  # 设置边框线宽为2.5

    # plt.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=20, ncol=min(len(group_types), 8), frameon=True, columnspacing=1.0, handletextpad=0.1, borderpad=0.5)

    # 保持 Audio Encoder 命名
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
    # output_dir = os.path.join(data_dir, "legend")
    os.makedirs(output_dir, exist_ok=True)

    # target_types_list = [["accent"], ["emphasis"], ["intonation"], ["noise"], ["speed"], ["tone"], ["emotion"], ["emphasis", "speed", "intonation", "tone", "noise", "accent", "emotion"]]
    target_types_list = [["emphasis", "speed", "intonation", "tone", "noise", "accent", "emotion"]]

    # Define which layers to visualize
    # target_layers = ["encoder", "layer_0", "layer_4", "layer_8", "layer_12", "layer_16", "layer_20", "layer_24", "layer_28", "layer_31"]
    # target_layers_minicpm = ["encoder", "layer_0", "layer_4", "layer_8", "layer_12", "layer_16", "layer_20", "layer_24", "layer_27"]
    target_layers = ["encoder", "layer_0", "layer_8", "layer_16", "layer_24", "layer_31"]
    target_layers_minicpm = ["encoder", "layer_0", "layer_8", "layer_16", "layer_24", "layer_27"]

    # methods = ["tsne", "pca", "umap"]
    methods = ["tsne"]

    for target_types in target_types_list:
        # 每组都需要和原始音频比较
        target_types.append("original")

        for method in methods:
            for model_name in model_names:
                current_target_layers = []
                # MiniCPM 层数和其它模型不一样
                if model_name == "MiniCPM-o-2_6":
                    current_target_layers = target_layers_minicpm
                else:
                    current_target_layers = target_layers

                for target_layer in current_target_layers:
                    # 获取目录下所有的 pkl 文件
                    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
                    
                    # 筛选符合条件的文件
                    matched_files = []
                    for target_type in target_types:
                        for pkl_file in pkl_files:
                            basename = os.path.basename(pkl_file)
                            if model_name in basename and target_type in basename:
                                matched_files.append(pkl_file)
                    
                    # 读取并合并所有匹配的文件内容
                    merged_embeddings = {}
                    for pkl_file in matched_files:
                        with open(pkl_file, 'rb') as f:
                            embeddings_dict = pickle.load(f)
                        for key, value in embeddings_dict.items():
                            if key not in merged_embeddings:
                                merged_embeddings[key] = {}
                            merged_embeddings[key].update(value)
                    
                    # Visualize for each specified layer
                    visualize_embeddings(merged_embeddings, method, model_name, output_dir, target_types, target_layer)