import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dccrn.dccrn_model import DCCRN
from dccrn.dataset import AudioDataset
from dccrn.losses import Losses
from dccrn.audio_utils import AudioUtils
import os
import argparse

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset & Loader
    dataset = AudioDataset(args.clean_dir, args.noise_dir, segment_length=args.segment_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 2. Model, Optimizer, Loss
    model = DCCRN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (noisy_wf, clean_wf) in enumerate(dataloader):
            noisy_wf, clean_wf = noisy_wf.to(device), clean_wf.to(device)
            
            # STFT
            noisy_spec = AudioUtils.stft(noisy_wf)
            # Re/Im stacking for model input
            freq_bins = 257
            real = noisy_spec[:, :freq_bins, :]
            imag = noisy_spec[:, freq_bins:, :]
            model_input = torch.stack([real, imag], dim=1)
            
            # Forward
            est_spec_stacked = model(model_input)
            
            # Reconstruct to [B, 2*F, T]
            est_spec = torch.cat([est_spec_stacked[:, 0, :, :], est_spec_stacked[:, 1, :, :]], dim=1)
            
            # Inverse STFT to get estimated waveform
            est_wf = AudioUtils.istft(est_spec, length=clean_wf.shape[-1])
            
            # Loss Calculation (SI-SNR on waveform)
            loss = Losses.si_snr(est_wf, clean_wf)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # Save Checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"dccrn_epoch_{epoch+1}.pt"))
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help="Path to clean speech dir")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to noise dir")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment_length", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    
    args = parser.parse_args()
    train(args)
