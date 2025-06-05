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

## 🔧 Installation Guide

```bash
# Clone repository
git clone https://github.com/Researchtopic/Code-Jailbreak-AudioBench
cd Code-Jailbreak-AudioBench

# Create conda environment
conda create -n audio_editing_jailbreak python=3.10
conda activate audio_editing_jailbreak
pip install -r requirements.txt

# Install dependencies
sudo apt-get update
sudo apt-get install ffmpeg
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

## 🔍 Code and Paper Correspondence

This codebase implements the complete experimental pipeline described in the paper:

1. **Audio Editing Toolbox** (Section 2) - Implemented in `Editing/`, supporting seven different types of audio editing operations.
2. **Dataset Creation** (Section 3) - The complete Jailbreak-AudioBench dataset is constructed using the tools in `Editing/`.
3. **Model Evaluation** (Section 3) - Evaluation of all involved LALM models is implemented in `Inference/`.
4. **Query-based Audio Editing Jailbreak Attack** (Section 4.1) - Implements the Query-based Audio Editing Jailbreak method by combining audio edits.
5. **Defense Method** (Section 4.2) - Evaluates basic defense capabilities by prepending a defense prompt.


## 📦 Pre-trained Models
This project uses the following third-party models:

* **BLSP** – [Github](https://github.com/cwang621/blsp)
* **SpeechGPT** – [Github](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt)
* **Qwen2-Audio-7B-Instruct** – [HuggingFace model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
* **SALMONN-7B** – [HuggingFace model card](https://huggingface.co/tsinghua-ee/SALMONN-7B)
* **SALMONN-13B** – [HuggingFace model card](https://huggingface.co/tsinghua-ee/SALMONN)
* **VITA-1.5** – [HuggingFace model card](https://huggingface.co/VITA-MLLM/VITA-1.5)
* **R1-AQA** – [HuggingFace model card](https://huggingface.co/mispeech/r1-aqa)
* **MiniCPM-o-2.6** – [HuggingFace model card](https://huggingface.co/openbmb/MiniCPM-o-2_6)


## 📜 Citation
If you use Jailbreak-AudioBench in the research, please cite our paper:

```bibtex
@misc{cheng2025jailbreakaudiobenchindepthevaluationanalysis,
      title={Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models}, 
      author={Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu},
      year={2025},
      eprint={2501.13772},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2501.13772}, 
}
```


## 📄 Licence
The code in this repository is released under the **MIT License**.  
Jailbreak prompts originate from public datasets (AdvBench, MM-SafetyBench, RedTeam-2K, SafeBench) and comply with their respective licences.
