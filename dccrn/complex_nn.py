import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super(ComplexConv2d, self).__init__()
        self.conv_re = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.conv_im = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        
    def forward(self, input):
        # input: [batch, 2*in_channels, h, w] -> Real is first half, Imag is second half
        in_re, in_im = torch.chunk(input, 2, dim=1)
        
        out_re = self.conv_re(in_re) - self.conv_im(in_im)
        out_im = self.conv_re(in_im) + self.conv_im(in_re)
        
        return torch.cat([out_re, out_im], dim=1)

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        super(ComplexConvTranspose2d, self).__init__()
        self.conv_re = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation)
        self.conv_im = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation)
        
    def forward(self, input):
        in_re, in_im = torch.chunk(input, 2, dim=1)
        
        out_re = self.conv_re(in_re) - self.conv_im(in_im)
        out_im = self.conv_re(in_im) + self.conv_im(in_re)
        
        return torch.cat([out_re, out_im], dim=1)

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        self.bn_re = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_im = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, input):
        in_re, in_im = torch.chunk(input, 2, dim=1)
        return torch.cat([self.bn_re(in_re), self.bn_im(in_im)], dim=1)

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, projection_dim=None, bidirectional=False):
        super(ComplexLSTM, self).__init__()
        self.lstm_re = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
    def forward(self, input):
        # input: [batch, seq_len, 2*input_size]
        in_re, in_im = torch.chunk(input, 2, dim=-1)
        
        # Batch-parallel optimization: 
        # Pack Real and Imag into the batch dimension to run them through the LSTM in one go.
        batch_size = in_re.shape[0]
        in_all = torch.cat([in_re, in_im], dim=0) # [2*B, T, D]
        
        out_re_all, _ = self.lstm_re(in_all) # [2*B, T, H]
        out_im_all, _ = self.lstm_im(in_all) # [2*B, T, H]
        
        out_re_re, out_re_im = torch.chunk(out_re_all, 2, dim=0)
        out_im_re, out_im_im = torch.chunk(out_im_all, 2, dim=0)
        
        out_re = out_re_re - out_im_im
        out_im = out_re_im + out_im_re
        
        return torch.cat([out_re, out_im], dim=-1)
