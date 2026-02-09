import torch
import torchaudio
import argparse
import os
from dccrn.dccrn_model import DCCRN
from dccrn.audio_utils import AudioUtils

from scipy.io import wavfile
import numpy as np

def denoise(input_path, output_path, model_path=None):
    # 1. Load Audio and Resample if needed
    sample_rate, data = wavfile.read(input_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    waveform = torch.from_numpy(data).float()
    
    if sample_rate != 16000:
        print(f"Warning: Input SR is {sample_rate}Hz. Resampling to 16000Hz for the model...")
        import torchaudio.transforms as T
        resampler = T.Resample(sample_rate, 16000)
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        waveform = resampler(waveform)
        sample_rate = 16000

    orig_len = waveform.shape[-1]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        if waveform.shape[0] > waveform.shape[1] and waveform.shape[1] < 10:
            waveform = waveform.T
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Transform to Spectrogram
    spec = AudioUtils.stft(waveform.to(device)) 
    
    freq_bins = 257
    real = spec[:, :freq_bins, :]
    imag = spec[:, freq_bins:, :]
    model_input = torch.stack([real, imag], dim=1) 
    
    # 3. Initialize Model
    model = DCCRN().to(device)
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Forward Pass with OVERLAPPING Chunking
    chunk_size = 1500 
    margin = 100
    num_frames = model_input.shape[3]
    denoised_spec_stacked = torch.zeros_like(model_input)
    
    with torch.no_grad():
        if num_frames > chunk_size:
            print(f"File is large ({num_frames} frames). Processing with Overlapping Context...")
            for i in range(0, num_frames, chunk_size):
                start = max(0, i - margin)
                end = min(i + chunk_size + margin, num_frames)
                
                chunk = model_input[:, :, :, start:end]
                denoised_full_chunk = model(chunk)
                
                # Calculate internal indices relative to the margin
                saved_start_in_chunk = i - start
                saved_end_in_chunk = min(i + chunk_size, num_frames) - start
                
                denoised_spec_stacked[:, :, :, i:min(i + chunk_size, num_frames)] = \
                    denoised_full_chunk[:, :, :, saved_start_in_chunk:saved_end_in_chunk]
        else:
            denoised_spec_stacked = model(model_input) 
    
    # 5. Inverse Transform
    denoised_spec = torch.cat([denoised_spec_stacked[:, 0, :, :], 
                               denoised_spec_stacked[:, 1, :, :]], dim=1)
    
    clean_waveform = AudioUtils.istft(denoised_spec, length=orig_len)
    
    # 6. Save using scipy
    clean_audio = clean_waveform.squeeze().cpu().numpy()
    
    # Noise Blending (Inject "Comfort Noise" to prevent chopping)
    alpha = 0.1
    orig_audio = waveform.squeeze().cpu().numpy()
    clean_audio = (1 - alpha) * clean_audio + alpha * orig_audio
    
    # Normalize
    max_val = np.max(np.abs(clean_audio))
    if max_val > 0:
        clean_audio = clean_audio / max_val
    wavfile.write(output_path, sample_rate, (clean_audio * 32767).astype(np.int16))
    print(f"Success! Denoised audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCCRN Denoising Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to noisy wav file")
    parser.add_argument("--output", type=str, default="denoised.wav", help="Path to save cleaned wav")
    parser.add_argument("--weights", type=str, help="Path to trained .pt weights")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        denoise(args.input, args.output, args.weights)
    else:
        print(f"Error: File not found: {args.input}")
