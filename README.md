# Noise Cancellation & Denoising Project

This repository contains two cutting-edge approaches to noise reduction: **Active Noise Cancellation (FxLMS)** for real-time acoustic interference and **Deep Learning Denoising (DCCRN)** for surgical digital audio cleaning.

## 📂 Project Structure

### 1. [FxLMS (Filtered-X Least Mean Squares)](./fxlms/)
A classic Digital Signal Processing (DSP) approach used in noise-canceling headphones. It targets predictable, tonal noise by generating "anti-noise" phase-inverted signals.
- **Key Files**: `anc.py` (Real-time engine), `optimized.py` (Best DSP parameters).
- **Ideal for**: Sustained engine hum, fan noise, and repetitive machinery.

### 2. [DCCRN (Deep Complex Convolutional Recurrent Network)](./dccrn/)
A state-of-the-art AI model that uses a U-Net style architecture with Complex-valued LSTMs to distinguish human speech from background noise.
- **Key Files**: `denoise.py` (Inference/Usage), `model.py` (Neural Architecture), `train.py` (Kaggle-ready Training).
- **Ideal for**: Sudden noises (clapping, barking), wind, and complex indoor backgrounds.

---

## 🚀 Quick Start

### AI Denoising (DCCRN)
To clean a noisy audio file using the trained model:
```bash
python dccrn/denoise.py --input noisy_audio.wav --output cleaned_audio.wav --weights dccrn/dccrn_e1.pt
```

### Active Noise Cancellation (FxLMS)
To simulate real-time noise cancellation:
```bash
python fxlms/anc.py
```

---

## 📊 Performance Comparison

| Feature | FxLMS (DSP) | DCCRN (AI) |
| :--- | :--- | :--- |
| **Strategy** | Phase-Inversion (Anti-noise) | High-dimensional Masking |
| **Complexity** | Extremely Low (Runs on tiny chips) | High (Needs GPU for training) |
| **Best For** | Predictable, Tonal noises | Unpredictable, Transient noises |
| **Voice Quality** | Natural (no distortions) | Surgical (may sound electronic if under-trained) |

---

## 📈 Visualizations
You can find performance plots (Noisy vs. Cleaned) in the respective `plots/` subdirectories. Use `dccrn/plot_performance.py` to generate new analysis for your audio files.

---

## 🧠 Training (Kaggle)
The DCCRN model was trained on the **LibriSpeech** and **MUSAN** datasets. A comprehensive [Kaggle Training Guide](./dccrn/KAGGLE_TRAINING_GUIDE.md) is included for those looking to reach higher SI-SNR levels (>20dB).
