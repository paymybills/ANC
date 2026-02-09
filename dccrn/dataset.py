import glob

class AudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, sample_rate=16000, segment_length=1.0, snr_range=(-5, 15)):
        self.clean_files = self._get_files(clean_dir)
        self.noise_files = self._get_files(noise_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.snr_range = snr_range

    def _get_files(self, directory):
        # Support recursive search and multiple formats
        formats = ['*.wav', '*.flac', '*.mp3']
        files = []
        for fmt in formats:
            files.extend(glob.glob(os.path.join(directory, "**", fmt), recursive=True))
        return sorted(files)

    def __len__(self):
        return len(self.clean_files)

    def _load_and_segment(self, path):
        try:
            # Try torchaudio if installed, otherwise soundfile/scipy
            import torchaudio
            data, sr = torchaudio.load(path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                data = resampler(data)
            data = data[0].numpy() # Mono 
        except Exception:
            try:
                import soundfile as sf
                data, sr = sf.read(path)
                if len(data.shape) > 1: data = data[:, 0]
            except Exception:
                sr, data = wavfile.read(path)
                data = data.astype(np.float32) / 32768.0

        if len(data) > self.segment_samples:
            start = random.randint(0, len(data) - self.segment_samples)
            data = data[start:start + self.segment_samples]
        else:
            data = np.pad(data, (0, self.segment_samples - len(data)))
        return torch.from_numpy(data).float()

    def _mix(self, clean, noise):
        # Calculate SNR
        snr = random.uniform(*self.snr_range)
        
        # Calculate power
        clean_pwr = torch.mean(clean ** 2) + 1e-8
        noise_pwr = torch.mean(noise ** 2) + 1e-8
        
        # Calculate gain for noise to reach desired SNR
        # SNR = 10 * log10(P_clean / P_noise)
        # P_noise_target = P_clean / (10^(SNR/10))
        gain = torch.sqrt(clean_pwr / (10 ** (snr / 10.0)) / noise_pwr)
        
        noisy = clean + gain * noise
        return noisy

    def __getitem__(self, idx):
        clean = self._load_and_segment(self.clean_files[idx])
        noise_idx = random.randint(0, len(self.noise_files) - 1)
        noise = self._load_and_segment(self.noise_files[noise_idx])
        
        noisy = self._mix(clean, noise)
        
        return noisy, clean
