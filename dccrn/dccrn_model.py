import torch
import torch.nn as nn
from .complex_nn import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexLSTM

class DCCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super(DCCRN, self).__init__()
        
        # Architecture parameters
        # encoder_channels: 1 (input) -> 32 -> 64 -> 128 -> 256 -> 256 -> 256
        self.encoder_channels = [1, 32, 64, 128, 256, 256, 256] 
        kernel_size = (5, 2)
        stride = (2, 1)
        padding = (2, 1)
        
        # 1. Encoder (6 Layers)
        self.encoder = nn.ModuleList()
        for i in range(len(self.encoder_channels) - 1):
            self.encoder.append(nn.Sequential(
                ComplexConv2d(self.encoder_channels[i], self.encoder_channels[i+1], kernel_size, stride, padding),
                ComplexBatchNorm2d(self.encoder_channels[i+1]),
                nn.PReLU()
            ))
            
        # 2. LSTM (2 Layers)
        # h=5 after 6 layers
        self.lstm_input_size = self.encoder_channels[-1] * 5
        self.lstm = ComplexLSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_input_size, num_layers=2)
        
        # 3. Decoder (6 Layers)
        # Decoder input channels: cat(previous_out, skip_connection)
        # E_outs: [32, 64, 128, 256, 256, 256] 
        # D1: LSTM(256) + E6(256) = 512 -> D1_out(256)
        # D2: D1_out(256) + E5(256) = 512 -> D2_out(256)
        # D3: D2_out(256) + E4(256) = 512 -> D3_out(128)
        # D4: D3_out(128) + E3(128) = 256 -> D4_out(64)
        # D5: D4_out(64) + E2(64) = 128 -> D5_out(32)
        # D6: D5_out(32) + E1(32) = 64 -> D6_out(1)
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
        # x: [batch, 2, freq, time] - Real/Imag in dim 1
        
        # Store encoder outputs for skip connections
        encoder_outputs = []
        out = x
        
        for layer in self.encoder:
            out = layer(out)
            encoder_outputs.append(out)
            
        # LSTM input processing
        batch, c_comp, h, w = out.size()
        c = c_comp // 2
        out_lstm_in = out.permute(0, 3, 1, 2).reshape(batch, w, -1)
        
        # LSTM
        out_lstm = self.lstm(out_lstm_in)
        
        # Reshape back to spectral domain
        out_re, out_im = torch.chunk(out_lstm, 2, dim=-1)
        out_re = out_re.reshape(batch, w, c, h).permute(0, 2, 3, 1)
        out_im = out_im.reshape(batch, w, c, h).permute(0, 2, 3, 1)
        out = torch.cat([out_re, out_im], dim=1)
        
        # Decoder with Skip Connections
        for i in range(len(self.decoder)):
            # Concatenate skip connection from same index (going backwards)
            skip = encoder_outputs[-(i+1)]
            out = torch.cat([out, skip], dim=1)
            out = self.decoder[i](out)
            
        return out
