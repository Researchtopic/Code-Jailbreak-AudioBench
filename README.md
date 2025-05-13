# Jailbreak-AudioBench  
### Exhaustive Evaluation & Analysis of Large **Audio**-Language Models under Jailbreak Threats

> **NeurIPS 2025 (submitted)**  
> This repository hosts the **official implementation, dataset-creation pipeline, and evaluation code** for our paper  
> *“Jailbreak-AudioBench: Exhaustive Evaluation and Analysis of Large Audio-Language Models under Jailbreak Threats”*.

<div align="center">
  <!-- GitHub renders PDFs as links; if you prefer inline, convert to PNG -->
  <img src="Figs/framework.pdf" width="85%">
  <br>
  <sub>Figure&nbsp;1 – End-to-end pipeline: harmful prompts → TTS audio → Audio-Editing Toolbox → Benchmark.</sub>
</div>

---

## ✨ Highlights
* **Audio Editing Toolbox (AET)** - seven edits (*Emphasis · Speed · Intonation · Tone · Background-Noise · Celebrity-Accent · Emotion*) implemented in pure Python/SOX under [`audio_shift_code/`](audio_shift_code/).
* **Jailbreak-AudioBench Dataset** - 4 700 base audios × 20 edits → **94 800** clips (+ Defense subset) covering easy/complex tasks.
* **Plug-and-play evaluation** for BLSP, SpeechGPT, Qwen2-Audio, SALMONN, etc., with automatic safety judgement via **Llama Guard 3**.
* **Greedy black-box attack search** plus rich visual analysis  
  <img src="Figs/tsne%20visualization.jpg" width="75%">

---

## 🔧 Requirements

```bash
conda create -n jab python=3.10
conda activate jab
pip install -r requirements.txt      # librosa, pydub, soundfile, torch, tqdm …
# SOX for pitch/tempo
sudo apt-get install sox libsox-fmt-all


