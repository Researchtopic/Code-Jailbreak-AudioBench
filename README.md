# Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models

<div align="left">
  <a href="https://researchtopic.github.io/Jailbreak-AudioBench_Page/" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-Jailbreak--AudioBench-blue?style=for-the-badge&logo=github" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/researchtopic/Jailbreak-AudioBench" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?style=for-the-badge&logo=huggingface" alt="Dataset">
  </a>
</div>

> This repository hosts the implementation of **dataset generation pipeline** and **evaluation code** for our paper *“Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models”*.

<div align="center">
  <img src="Figs/framework.png" width="90%">
  <br>
  <sub>Figure 1 –– Pipeline: harmful prompts → TTS audio → Audio-Editing Toolbox → Benchmark.</sub>
  <br>
</div>

## ✨ Highlights
* **Audio Editing Toolbox (AET)** ––– seven audio editings (*Emphasis · Speed · Intonation · Tone · Background Noise · Celebrity Accent · Emotion*) implemented in Python under `Editing/`.

<div align="center">
  <img src="Figs/audio_editing.png" width="90%">
  <br>
  <sub>Figure 2 –– Examples of injecting different audio hidden semantics.</sub>
  <br>
</div>

* **Jailbreak-AudioBench Dataset** ––– 4,700 base audios × 20 editing types = **94,800** audio samples covering explicit and implicit jailbreak tasks. The dataset also includes an equal number of defended versions of these audio samples to explore defense strategies against audio editing jailbreaks.
* **Plug-and-play evaluation** for various Large Audio Language Models (LALMs) with automatic safety judgement via **Llama Guard 3**.
* **Query-based Audio Editing Jailbreak Method** combining different audio editing types, achieve higher Attack Success Rate (ASR) on State-of-the-Art LALMs.

<div align="center">
  <img src="Figs/grid search heatmap.png" width="90%">
  <br>
  <sub>Figure 3 –– ASR performance of Query-based Audio Editing Jailbreak.</sub>
</div>

## 🔧 Requirements

```bash
# create environment
conda create -n jab python=3.10
conda activate jab

# python deps
pip install -r requirements.txt          # librosa, pydub, soundfile, torch, tqdm …

# SOX for pitch / tempo
sudo apt-get install sox libsox-fmt-all
```

## 🗂️ Directory Layout
```text
├── Editing/                        # dataset generation code
│   ├── audio_shift_original.py     # original audio generation
│   ├── audio_shift_tone.py         
│   ├── audio_shift_speed.py        
│   ├── audio_shift_emphasis.py     
│   ├── audio_shift_intonation.py   
│   ├── audio_noise.py              
│   ├── audio_noise_crowd.py        
│   └── combine.py                  
├── Inference/                      # model inference code
│   ├── BLSP.py                     
│   ├── VITA1.5.py                  
│   ├── gpt4o.py                    
│   ├── qwen2_audio.py              
│   ├── salmonn_13b.py              
│   └── speechgpt.py                
├── Figs/                           # paper figures & visualisations
└── README.md
```

## 🏗️ Dataset Generation

```bash
# 1️⃣ text → base audios (16 kHz)
python Editing/audio_shift_original.py

# 2️⃣ resample (optional)
python Editing/convert_sampling_rate.py

# 3️⃣ example edit: Tone +4 semitones
python Editing/audio_shift_tone.py

```

## 🏃‍♂️ Evaluation

```bash
# 1️⃣ example evaluation: MiniCPM-o-2.6
python Inference/minicpm-o-2.6.py

# 2️⃣ use Llama Guard 3 to judge whether the jailbreak is successful
python Inference/analysis/llama3_guard.py
```

## 📈 Key Results (Explicit Subtype)

| Model | Original | Tone –8 | Tone +8 | Speed ×1.5 | Crowd Noise | **Worst Δ ↑** |
|-------|---------:|--------:|--------:|-----------:|------------:|--------------:|
| **BLSP**            | 47.5% | 44.4% | 45.1% | **44.9% ↓** | 48.3% | – 2.6% |
| **SpeechGPT**       | 14.1% | 10.2% | 0.5% | 14.3% | **7.6% ↓** | – 13.6% |
| **Qwen2-Audio-7B**  | 16.8% | 11.7% | **13.6% ↓** | 17.9% | 9.1% | – 7.7% |
| **SALMONN-13B**     | 31.3% | **42.8% ↑** | **55.4% ↑** | **8.4% ↓** | **58.9% ↑** | + 27.6% |

*Full tables & t-SNE figures are available in the `Inference/analysis/` directory.*

## 🔍 Code and Paper Correspondence

This codebase implements the complete experimental pipeline described in the paper:

1. **Audio Editing Toolbox** (Section 2) - Implemented in `Editing/`, supporting seven different types of audio editing operations.
2. **Dataset Creation** (Section 3) - The complete Jailbreak-AudioBench dataset is constructed using the tools in `Editing/`.
3. **Model Evaluation** (Section 3) - Evaluation of all involved LALM models is implemented in `Inference/`.
4. **Query-based Audio Editing Jailbreak Attack** (Section 4.1) - Implements the Query-based Audio Editing Jailbreak method by combining audio edits.
5. **Defense Method** (Section 4.2) - Evaluates basic defense capabilities by prepending a defense prompt.


## ✅ Code Completeness Checklist
- [x] **Dependencies** (`requirements.txt`, conda, SOX install)
- [x] **Dataset Generation Code** (`Editing/`)
- [x] **Evaluation code** (`Inference/`)


## 📦 Pre-trained Models
This project uses the following third-party models:

* **Qwen2-Audio-7B-Instruct** – [HuggingFace model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
* **MiniCPM-o-2.6** – [HuggingFace model card](https://huggingface.co/openbmb/MiniCPM-o-2_6)


## 📜 Citation
If you use Jailbreak-AudioBench in your research, please cite our paper:

```bibtex
@misc{cheng2025jailbreakaudiobenchindepthevaluationanalysis,
      title={Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models}, 
      author={Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Sheng, Philip Torr, Jindong Gu, Renjing Xu},
      year={2025},
      eprint={2501.13772},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2501.13772}, 
}
```


## 📄 Licence
The code in this repository is released under the **MIT License**.  
Jailbreak prompts originate from public datasets (AdvBench, MM-SafetyBench, RedTeam-2K, SafeBench) and comply with their respective licences.
