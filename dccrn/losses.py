import torch

class Losses:
    @staticmethod
    def si_snr(est, target):
        """
        Scale-Invariant Signal-to-Noise Ratio
        est: [Batch, Samples]
        target: [Batch, Samples]
        """
        eps = 1e-8
        # Ensure mean-zero for scale invariance
        est = est - torch.mean(est, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)
        
        # Target component
        # s_target = (<est, target> * target) / ||target||^2
        dot_prod = torch.sum(est * target, dim=-1, keepdim=True)
        target_norm = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
        s_target = (dot_prod / target_norm) * target
        
        # Error component
        e_noise = est - s_target
        
        # Calculation
        target_pwr = torch.sum(s_target ** 2, dim=-1) + eps
        noise_pwr = torch.sum(e_noise ** 2, dim=-1) + eps
        
        snr = 10 * torch.log10(target_pwr / noise_pwr)
        return -torch.mean(snr) # Negative SNR for minimization

    @staticmethod
    def complex_mse(est_spec, target_spec):
        # est_spec: [B, 2*F, T] (DCCRN Output)
        return torch.mean((est_spec - target_spec) ** 2)
