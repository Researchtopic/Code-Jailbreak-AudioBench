# Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models

<div align="left">
  <a href="https://researchtopic.github.io/Jailbreak-AudioBench_Page/" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-Jailbreak--AudioBench-blue?style=for-the-badge&logo=github" alt="Project Page">
  </a>
  <a href="https://huggingface.co/datasets/researchtopic/Jailbreak-AudioBench" target="_blank">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow?style=for-the-badge&logo=huggingface" alt="Dataset">
  </a>
</div>

> This repository hosts the **official implementation, dataset-creation pipeline, and evaluation code** for our paper  
> *“Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models”*.

<div align="center">
  <img src="Figs/framework.png" width="85%">
  <br>
  <sub>Figure 1 – End-to-end pipeline: harmful prompts → TTS audio → Audio-Editing Toolbox → Benchmark.</sub>
</div>

---

## ✨ Highlights
* **Audio Editing Toolbox (AET)** – seven edits (*Emphasis · Speed · Intonation · Tone · Background-Noise · Celebrity-Accent · Emotion*) implemented in pure Python + SOX under [`audio_shift_code`](audio_shift_code/).

<div align="center">
  <img src="Figs/audio_editing.png" width="85%">
  <br>
  <sub>Figure 2 – Examples of injecting different audio hidden semantics.</sub>
</div>

* **Jailbreak-AudioBench Dataset** – 4,700 base audios × 20 edit types → **94,800** clips (+ Defense subset) covering explicit and implicit jailbreak tasks.
* **Plug-and-play evaluation** for BLSP, SpeechGPT, Qwen2-Audio, SALMONN, … with automatic safety judgement via **Llama Guard 3**.
* **Query-based Audio Editing Jailbreak Method** plus rich visuals  

<div align="center">
  <img src="Figs/grid search heatmap.png" width="85%">
  <br>
  <sub>Figure 3 – Query-based Audio Editing Jailbreak Attack Success Rate - even GPT-4o-Audio shows vulnerability.</sub>
</div>

---

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

---

## 🗂️ Directory Layout
```text
├── audio_shift_code/          # data-creation / editing scripts
│   ├── audio_shift_original.py  # original audio generation
│   ├── audio_shift_tone.py      # tone adjustment
│   ├── audio_shift_speed.py     # speed adjustment
│   ├── audio_shift_emphasis.py  # emphasis processing
│   ├── audio_shift_intonation.py # intonation adjustment
│   ├── audio_noise.py           # noise addition
│   ├── audio_noise_crowd.py     # crowd noise
│   └── combine.py               # combination processing
├── Inference/                  # model inference code
│   ├── BLSP.py                  # BLSP model evaluation
│   ├── VITA1.5.py               # VITA-1.5 model evaluation
│   ├── gpt4o.py                 # GPT-4o-Audio evaluation
│   ├── qwen2_audio.py           # Qwen2-Audio evaluation
│   ├── salmonn_13b.py           # SALMONN-13B evaluation
│   └── speechgpt.py             # SpeechGPT evaluation
├── Figs/                      # paper figures & visualisations
└── README.md
```

---

## 🏗️ Dataset Generation

```bash
# 1️⃣  text → base audios (16 kHz)
python audio_shift_code/audio_shift_original.py \
       --csv data/jailbreak_questions.csv \
       --out_dir data/base_audio

# 2️⃣  resample (optional)
python audio_shift_code/convert_sampling_rate.py \
       --in_dir data/base_audio --sr 16000

# 3️⃣  example edit: Tone +4 semitones
python audio_shift_code/audio_shift_tone.py \
       --in_dir data/base_audio --out_dir data/tone/+4 --n_steps 4

# 4️⃣  run *all* edits
bash scripts/run_all_edits.sh data/base_audio data/edited
```

---

## 🏃‍♂️ Evaluation

```bash
python Inference/eval_lalm.py \
       --model qwen2-audio-7b \
       --split explicit \
       --audio_dir data/edited/tone/+4 \
       --judge llama-guard-3 \
       --save_csv results/qwen2_tone+4.csv
```

The script feeds each audio to the chosen LALM, lets **Llama Guard 3** label the answer, and stores **Attack Success Rate (ASR)** plus raw generations.

---

## 📈 Key Results (Explicit subset)

| Model | Original | Tone –8 | Tone +8 | Speed ×1.5 | Crowd Noise | **Worst Δ ↑** |
|-------|---------:|--------:|--------:|-----------:|------------:|--------------:|
| **BLSP**            | 47.5% | 44.4% | 45.1% | **44.9% ↓** | 48.3% | – 2.6% |
| **SpeechGPT**       | 14.1% | 10.2% | 0.5% | 14.3% | **7.6% ↓** | – 13.6% |
| **Qwen2-Audio-7B**  | 16.8% | 11.7% | **13.6% ↓** | 17.9% | 9.1% | – 7.7% |
| **SALMONN-13B**     | 31.3% | **42.8% ↑** | **55.4% ↑** | **8.4% ↓** | **58.9% ↑** | + 27.6% |

*Full tables & t-SNE figures are available in the `Inference/analysis/` directory.*

---

## 🔍 Code and Paper Correspondence

This codebase implements the complete experimental pipeline described in the paper:

1. **Audio Editing Toolbox** (Section 2) - Implemented in `audio_shift_code`, supporting seven different types of audio editing operations.
2. **Dataset Creation** (Section 3) - The complete Jailbreak-AudioBench dataset is constructed using the tools in `audio_shift_code`.
3. **Model Evaluation** (Section 3) - Evaluation of all involved LALM models is implemented in `Inference`.
4. **Query-based Audio Editing Jailbreak Attack** (Section 4.1) - Implements the Query-based Audio Editing Jailbreak method by combining audio edits.
5. **Defense Method** (Section 4.2) - Evaluates basic defense capabilities by prepending a defense prompt.

---

## 📦 Pre-trained Checkpoints
This project uses the following third-party models:

* **BLSP** – [HuggingFace model card](https://huggingface.co/Bluster/Bluster-TTS-v0.1)
* **SALMONN-13B** – [HuggingFace model card](https://huggingface.co/SALMONN/SALMONN-13B)

---

## 📑 Data and Resources

<div align="left">
  <a href="https://huggingface.co/datasets/researchtopic/Jailbreak-AudioBench" target="_blank">
    <img src="https://img.shields.io/badge/Full%20Dataset-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface" alt="Full Dataset">
  </a>
  <a href="https://researchtopic.github.io/Jailbreak-AudioBench_Page/" target="_blank">
    <img src="https://img.shields.io/badge/More%20Details-Project%20Page-blue?style=for-the-badge&logo=github" alt="More Details">
  </a>
</div>

We provide the following resources:
- **Full Dataset** - Contains all original and edited audio samples.
- **Editing Toolbox** - Allows researchers to create their own edited versions.
- **Evaluation Code** - A complete framework for LALM safety evaluation.

---

## ✅ ML Code Completeness Checklist
- [x] **Dependencies** (`requirements.txt`, conda, SOX install)
- [x] **Training / data-creation code** (`audio_shift_code/`)
- [x] **Evaluation code** (`Inference/`)  
- [x] **Pre-trained model links**
- [x] **README** with results + exact reproduction commands

---

## 📜 Citation
If you use Jailbreak-AudioBench in your research, please cite our paper:

```bibtex
@misc{cheng2025jailbreakaudiobenchindepthevaluationanalysis,
      title={Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models}, 
      author={Hao Cheng and Erjia Xiao and Jing Shao and Yichi Wang and Le Yang and Chao Sheng and Philip Torr and Jindong Gu and Renjing Xu},
      year={2025},
      eprint={2501.13772},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2501.13772}, 
}
```

---

## 📄 Licence
The code in this repository is released under the **MIT License**.  
Jailbreak prompts originate from public datasets (AdvBench, MM-SafetyBench, RedTeam-2K, SafeBench) and comply with their respective licences.
