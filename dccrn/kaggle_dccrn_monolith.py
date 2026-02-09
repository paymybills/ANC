import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from scipy.io import wavfile
import argparse
from tqdm.auto import tqdm

# ==========================================
# 1. Complex Neural Network Modules
# ==========================================

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ComplexConv2d, self).__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x: [batch, 2*in_channels, h, w] -> Real/Imag stacked in dim 1
        x_re, x_im = torch.chunk(x, 2, dim=1)
        out_re = self.conv_re(x_re) - self.conv_im(x_im)
        out_im = self.conv_re(x_im) + self.conv_im(x_re)
        return torch.cat([out_re, out_im], dim=1)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_re = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_im = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        x_re, x_im = torch.chunk(x, 2, dim=1)
        out_re = self.conv_re(x_re) - self.conv_im(x_im)
        out_im = self.conv_re(x_im) + self.conv_im(x_re)
        return torch.cat([out_re, out_im], dim=1)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_re = nn.BatchNorm2d(num_features)
        self.bn_im = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x_re, x_im = torch.chunk(x, 2, dim=1)
        return torch.cat([self.bn_re(x_re), self.bn_im(x_im)], dim=1)

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ComplexLSTM, self).__init__()
        self.lstm_re = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm_im = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: [batch, time, 2*input_size]
        x_re, x_im = torch.chunk(x, 2, dim=-1)
        out_re_re, _ = self.lstm_re(x_re)
        out_re_im, _ = self.lstm_re(x_im)
        out_im_re, _ = self.lstm_im(x_re)
        out_im_im, _ = self.lstm_im(x_im)
        out_re = out_re_re - out_im_im
        out_im = out_re_im + out_im_re
        return torch.cat([out_re, out_im], dim=-1)

# ==========================================
# 2. Audio Utilities (STFT/iSTFT/Compression)
# ==========================================

class AudioUtils:
    @staticmethod
    def stft(waveform, n_fft=512, hop_length=256):
        window = torch.hann_window(n_fft).to(waveform.device)
        spec = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                          win_length=n_fft, window=window, center=True, 
                          return_complex=True)
        # Apply Power Compression (0.3)
        mag = torch.abs(spec)
        phase = torch.angle(spec)
        mag = mag ** 0.3
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.cat([real, imag], dim=1)

    @staticmethod
    def istft(spec, n_fft=512, hop_length=256, length=None):
        # spec: [batch, 2*freq, time]
        freq_bins = n_fft // 2 + 1
        real = spec[:, :freq_bins, :]
        imag = spec[:, freq_bins:, :]
        # Decompression (1/0.3)
        complex_spec = torch.complex(real, imag)
        mag = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)
        mag = mag ** (1.0 / 0.3)
        complex_spec = torch.polar(mag, phase)
        
        window = torch.hann_window(n_fft).to(spec.device)
        return torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length,
                           win_length=n_fft, window=window, center=True, length=length)

# ==========================================
# 3. DCCRN Model
# ==========================================

class DCCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(DCCRN, self).__init__()
        self.encoder_channels = [1, 32, 64, 128, 256, 256, 256] 
        kernel_size = (5, 2)
        stride = (2, 1)
        padding = (2, 1)
        
        self.encoder = nn.ModuleList()
        for i in range(len(self.encoder_channels) - 1):
            self.encoder.append(nn.Sequential(
                ComplexConv2d(self.encoder_channels[i], self.encoder_channels[i+1], kernel_size, stride, padding),
                ComplexBatchNorm2d(self.encoder_channels[i+1]),
                nn.PReLU()
            ))
            
        self.lstm_input_size = self.encoder_channels[-1] * 5
        self.lstm = ComplexLSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_input_size, num_layers=2)
        
        self.decode_in_channels = [512, 512, 512, 256, 128, 64]
        self.decode_out_channels = [256, 256, 128, 64, 32, 1]
        
        self.decoder = nn.ModuleList()
        for i in range(len(self.decode_in_channels)):
            self.decoder.append(nn.Sequential(
                ComplexConvTranspose2d(self.decode_in_channels[i], self.decode_out_channels[i], 
                                        kernel_size, stride, padding),
                ComplexBatchNorm2d(self.decode_out_channels[i]) if i < 5 else nn.Identity(),
                nn.PReLU() if i < 5 else nn.Identity()
            ))

    def forward(self, x):
        encoder_outputs = []
        out = x
        for layer in self.encoder:
            out = layer(out)
            encoder_outputs.append(out)
            
        batch, c_comp, h, w = out.size()
        c = c_comp // 2
        out_lstm_in = out.permute(0, 3, 1, 2).reshape(batch, w, -1)
        out_lstm = self.lstm(out_lstm_in)
        
        out_re, out_im = torch.chunk(out_lstm, 2, dim=-1)
        out_re = out_re.reshape(batch, w, c, h).permute(0, 2, 3, 1)
        out_im = out_im.reshape(batch, w, c, h).permute(0, 2, 3, 1)
        out = torch.cat([out_re, out_im], dim=1)
        
        for i in range(len(self.decoder)):
            skip = encoder_outputs[-(i+1)]
            out = torch.cat([out, skip], dim=1)
            out = self.decoder[i](out)
        return out

# ==========================================
# 4. Dataset & Loss
# ==========================================

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sample_rate=16000, segment_length=1.0, snr_range=(-5, 15)):
        self.clean_files = self._get_files(clean_dir)
        self.noise_files = self._get_files(noise_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.snr_range = snr_range

    def _get_files(self, directory):
        formats = ['*.wav', '*.flac', '*.mp3']
        files = []
        for fmt in formats:
            files.extend(glob.glob(os.path.join(directory, "**", fmt), recursive=True))
        return sorted(files)

    def __len__(self): return len(self.clean_files)

    def _load_and_segment(self, path):
        try:
            import torchaudio
            data, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                data = resampler(data)
            data = data[0].numpy()
        except:
            sr, data = wavfile.read(path)
            data = data.astype(np.float32) / 32768.0
        if len(data) > self.segment_samples:
            start = random.randint(0, len(data) - self.segment_samples)
            data = data[start:start + self.segment_samples]
        else:
            data = np.pad(data, (0, self.segment_samples - len(data)))
        return torch.from_numpy(data).float()

    def __getitem__(self, idx):
        clean = self._load_and_segment(self.clean_files[idx])
        noise = self._load_and_segment(self.noise_files[random.randint(0, len(self.noise_files)-1)])
        # Mix with SNR
        snr = random.uniform(*self.snr_range)
        c_pwr = torch.mean(clean**2) + 1e-8
        n_pwr = torch.mean(noise**2) + 1e-8
        gain = torch.sqrt(c_pwr / (10**(snr/10.0)) / n_pwr)
        noisy = clean + gain * noise
        return noisy, clean

def si_snr_loss(est, target):
    eps = 1e-8
    est = est - torch.mean(est, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    dot = torch.sum(est * target, dim=-1, keepdim=True)
    t_norm = torch.sum(target**2, dim=-1, keepdim=True) + eps
    s_target = (dot / t_norm) * target
    e_noise = est - s_target
    snr = 10 * torch.log10((torch.sum(s_target**2, dim=-1) + eps) / (torch.sum(e_noise**2, dim=-1) + eps))
    return -torch.mean(snr)

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, default="/kaggle/input/librispeech-asr-corpus/train-clean-360")
    parser.add_argument("--noise_dir", type=str, default="/kaggle/input/musan-dataset/musan/noise")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000, help="Limit steps per epoch for faster saves")
    
    # Use parse_known_args to avoid crashing in Jupyter/Kaggle notebooks
    args, _ = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DCCRN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = AudioDataset(args.clean_dir, args.noise_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    pbar_epoch = tqdm(range(args.epochs), desc="Training Phase")
    for epoch in pbar_epoch:
        model.train()
        total_loss = 0
        pbar_step = tqdm(loader, desc=f"Epoch {epoch}", leave=False, total=min(len(loader), args.max_steps_per_epoch))
        for i, (n_wf, c_wf) in enumerate(pbar_step):
            if i >= args.max_steps_per_epoch:
                break
            n_wf, c_wf = n_wf.to(device), c_wf.to(device)
            n_spec = AudioUtils.stft(n_wf)
            # Input shape: [B, 2, F, T]
            model_in = torch.stack([n_spec[:, :257, :], n_spec[:, 257:, :]], dim=1)
            est_spec_st = model(model_in)
            est_spec = torch.cat([est_spec_st[:,0,:,:], est_spec_st[:,1,:,:]], dim=1)
            est_wf = AudioUtils.istft(est_spec, length=c_wf.shape[-1])
            loss = si_snr_loss(est_wf, c_wf)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                pbar_step.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(loader)
        pbar_epoch.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
        torch.save(model.state_dict(), f"dccrn_e{epoch}.pt")
