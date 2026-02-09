import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.io import wavfile
import os

def plot_dccrn_performance(noisy_path, cleaned_path, save_path="dccrn_performance.png"):
    # 1. Load files
    fs_n, noisy = wavfile.read(noisy_path)
    fs_c, cleaned = wavfile.read(cleaned_path)
    
    # Normalize for plotting
    noisy = noisy.astype(np.float32) / (np.max(np.abs(noisy)) + 1e-8)
    cleaned = cleaned.astype(np.float32) / (np.max(np.abs(cleaned)) + 1e-8)
    
    # 2. Resample noisy to match cleaned length if SR is different
    if fs_n != fs_c:
        print(f"Resampling noisy input ({fs_n}Hz) to match cleaned output ({fs_c}Hz) for visualization...")
        from scipy import signal
        num_samples = len(cleaned)
        noisy = signal.resample(noisy, num_samples)
        fs_n = fs_c

    # 3. Setup Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Waveform Plot
    time = np.linspace(0, len(noisy)/fs_n, num=len(noisy))
    axes[0].plot(time, noisy, color='red', alpha=0.5, label="Noisy Input")
    axes[0].plot(time, cleaned, color='green', alpha=0.9, label="DCCRN Cleaned")
    axes[0].set_title("Waveform Comparison: Noisy vs DCCRN Cleaned")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)
    
    # Spectrogram Plot (Cleaned)
    axes[1].specgram(cleaned, Fs=fs_c, NFFT=512, noverlap=256, cmap='viridis')
    axes[1].set_title("Cleaned Spectrogram (Post-Processing)")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlabel("Time [sec]")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Check root and test_results fallback
    noisy = "my_audio.wav"
    cleaned = "cleaned_audio_v2.wav"
    
    if not os.path.exists(noisy):
        noisy = os.path.join("test_results", noisy)
    if not os.path.exists(cleaned):
        cleaned = os.path.join("test_results", cleaned)
        
    if os.path.exists(noisy) and os.path.exists(cleaned):
        plot_dccrn_performance(noisy, cleaned)
    else:
        print(f"File mismatch: ensure {noisy} and {cleaned} exist.")
