import torch
import torch.nn as nn
import torchaudio

class AudioUtils:
    @staticmethod
    def stft(audio, n_fft=512, hop_length=256, win_length=512):
        """
        Converts time-domain audio to complex spectrogram with Power Compression.
        """
        window = torch.hann_window(win_length).to(audio.device)
        spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                          win_length=win_length, window=window, 
                          center=True, return_complex=True)
        
        # Apply Power Compression (0.3) - CRITICAL for matching training
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag = mag ** 0.3
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        
        return torch.cat([real, imag], dim=1)

    @staticmethod
    def istft(complex_spec, n_fft=512, hop_length=256, win_length=512, length=None):
        """
        Converts complex spectrogram back to time-domain audio with Decompression.
        """
        freq_bins = complex_spec.shape[1] // 2
        real = complex_spec[:, :freq_bins, :]
        imag = complex_spec[:, freq_bins:, :]
        
        # Decompression (1/0.3) - CRITICAL for matching training
        c_spec = torch.complex(real, imag)
        mag = torch.abs(c_spec)
        phase = torch.angle(c_spec)
        mag = mag ** (1.0 / 0.3)
        c_spec = torch.polar(mag, phase)
        
        window = torch.hann_window(win_length).to(complex_spec.device)
        return torch.istft(c_spec, n_fft=n_fft, hop_length=hop_length, 
                           win_length=win_length, window=window, center=True, length=length)

    @staticmethod
    def power_compression(spec, alpha=0.3):
        """
        Applies power law compression to the magnitude of the spectrogram.
        Often used in DCCRN to match human hearing and stabilize training.
        """
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        
        compressed_mag = mag ** alpha
        return torch.complex(compressed_mag * torch.cos(phase), 
                             compressed_mag * torch.sin(phase))

    @staticmethod
    def power_decompression(compressed_spec, alpha=0.3):
        """
        Inverse of power_compression.
        """
        mag = torch.abs(compressed_spec)
        phase = torch.angle(compressed_spec)
        
        decompressed_mag = mag ** (1/alpha)
        return torch.complex(decompressed_mag * torch.cos(phase), 
                             decompressed_mag * torch.sin(phase))
