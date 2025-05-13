# Jailbreak-AudioBench  
### Exhaustive Evaluation & Analysis of Large **Audio**-Language Models under Jailbreak Threats

> **NeurIPS 2025 (submitted)**  
> This repository hosts the **official implementation, dataset-creation pipeline, and evaluation code** for our paper  
> *“Jailbreak-AudioBench: Exhaustive Evaluation and Analysis of Large Audio-Language Models under Jailbreak Threats”*.

<div align="center">
  <!-- GitHub renders PDFs as links; convert to PNG if you prefer inline -->
  <img src="Figs/framework.png" width="85%">
  <br>
  <sub>Figure 1 – End-to-end pipeline: harmful prompts → TTS audio → Audio-Editing Toolbox → Benchmark.</sub>
</div>

---

## ✨ Highlights
* **Audio Editing Toolbox (AET)** – seven edits (*Emphasis · Speed · Intonation · Tone · Background-Noise · Celebrity-Accent · Emotion*) implemented in pure Python + SOX under [`audio_shift_code/`](audio_shift_code/).
* **Jailbreak-AudioBench Dataset** – 4 700 base audios × 20 edits → **94 800** clips (+ Defense subset) covering easy / complex tasks.
* **Plug-and-play evaluation** for BLSP, SpeechGPT, Qwen2-Audio, SALMONN, … with automatic safety judgement via **Llama Guard 3**.
* **Greedy black-box attack search** plus rich visuals  
  <img src="Figs/tsne%20visualization.jpg" width="75%">

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
│   ├── audio_shift_original.py
│   ├── audio_shift_tone.py
│   ├── audio_shift_speed.py
│   ├── audio_shift_emphasis.py
│   ├── audio_shift_intonation.py
│   ├── audio_noise.py
│   ├── audio_noise_crowd.py
│   └── combine.py
├── Figs/                      # paper figures & visualisations
├── notebooks/                 # analysis & plotting
├── results/                   # CSVs, tables, reproduced figures
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
python eval/eval_lalm.py \
       --model qwen2-audio-7b \
       --split easy \
       --audio_dir data/edited/tone/+4 \
       --judge llama-guard-3 \
       --save_csv results/qwen2_tone+4.csv
```

The script feeds each audio to the chosen LALM, lets **Llama Guard 3** label the answer, and stores **Attack Success Rate (ASR)** plus raw generations.

---

## 📈 Key Results (AdvBench subset)

| Model | Original | Tone –8 | Tone +8 | Speed ×1.5 | Crowd Noise | **Worst Δ ↑** |
|-------|---------:|--------:|--------:|-----------:|------------:|--------------:|
| **BLSP**            | 0.598 | 0.523 | 0.508 | **0.486 ↓** | 0.565 | – 11.2 % |
| **SpeechGPT**       | 0.025 | 0.011 | 0.000 | 0.004 | **0.033 ↑** | + 0.8 % |
| **Qwen2-Audio-7B**  | 0.064 | 0.058 | **0.079 ↑** | 0.046 | 0.038 | + 1.5 % |
| **SALMONN-13B**     | 0.148 | **0.373 ↑** | **0.435 ↑** | **0.413 ↑** | **0.594 ↑** | + 44.6 % |

*Full tables & t-SNE figures are available in the `results/` and `Figs/` folders.*

---

## 📦 Pre-trained Checkpoints
To respect original licences we only link to external weights:

* **BLSP** – HuggingFace model card  
* **SALMONN-13B** – HuggingFace model card  
* Our finetuned **MiniCPM-o** defensive weights – [Zenodo DOI](https://doi.org/10.5281/zenodo.XXXXXX)

---

## 📜 Citation
```bibtex
@article{wang2025jailbreakaudiobench,
  title   = {Jailbreak-AudioBench: Exhaustive Evaluation and Analysis of Large Audio Language Models under Jailbreak Threats},
  author  = {Firstname Lastname and ...},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2025}
}
```

---

## ✅ ML Code Completeness Checklist
- [x] **Dependencies** (`requirements.txt`, conda, SOX install)
- [x] **Training / data-creation code** (`audio_shift_code/`)
- [x] **Evaluation code** (`eval/`)  
- [x] **Pre-trained model links**
- [x] **README** with results + exact reproduction commands

---

## 🤝 Contributing
Pull requests, issues and feature suggestions are welcome!  
Please read **`CONTRIBUTING.md`** and abide by the MIT licence.

---

## 📄 Licence
The code in this repository is released under the **MIT License**.  
Jailbreak prompts originate from public datasets (AdvBench, MM-SafetyBench, RedTeam-2K, SafeBench) and comply with their respective licences.
