import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from datetime import datetime

# -------------------------------
# OPTIMIZED FxLMS ANC - REAL-TIME
# -------------------------------
fs = 16000
N = 512
mu = 0.005
frame_len = 512

# Realistic secondary path S(z)
# In real application, this is measured via system identification
S_path = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1])
S_path = S_path / np.sum(np.abs(S_path))
L_s = len(S_path)

# S_hat is the estimate of S_path used for Filtered-X
S_hat = S_path.copy()

# Initialize weights and buffers
w = np.zeros(N)
x_buf = np.zeros(N)          # Reference signal buffer
y_buf = np.zeros(L_s)        # Anti-noise buffer for secondary path modeling
xf_buf = np.zeros(N)         # Filtered-X signal buffer

# Storage
input_audio = []
output_audio = []
error_audio = []
recording = True

print("=" * 70)
print("REAL-TIME ACTIVE NOISE CANCELLATION (Optimized)")
print("=" * 70)
print(f"Sample Rate: {fs} Hz | Filter: {N} taps | Learning: {mu}")
print("\nRunning... Press Ctrl+C to stop and save.")
print("=" * 70)

def callback(indata, outdata, frames, time, status):
    global w, x_buf, y_buf, xf_buf, input_audio, output_audio, error_audio, recording
    
    if not recording:
        outdata[:] = 0
        return
    
    x = indata[:, 0]
    y_out = np.zeros_like(x)
    e_out = np.zeros_like(x)

    for n in range(len(x)):
        # Update reference buffer (x[n] is current sample)
        x_buf = np.roll(x_buf, 1)
        x_buf[0] = x[n]
        
        # 1. Generate anti-noise
        y = np.dot(w, x_buf)
        y_out[n] = y
        
        # 2. Simulate secondary path (y filtered by S_path)
        y_buf = np.roll(y_buf, 1)
        y_buf[0] = y
        y_s = np.dot(S_path, y_buf)
        
        # 3. Error signal (disturbance x[n] minus secondary-path anti-noise)
        # Note: In real setup, 'e' is measured by a physical microphone
        e = x[n] - y_s
        e_out[n] = e
        
        # 4. Filtered-X: x filtered by S_hat
        # We only need the latest filtered sample for our Filtered-X buffer
        x_latest_s = np.dot(S_hat, x_buf[:L_s])
        xf_buf = np.roll(xf_buf, 1)
        xf_buf[0] = x_latest_s
        
        # 5. LMS Weight Update
        w = w + 2 * mu * e * xf_buf
        
        # Stability: Weight Clipping
        w_norm = np.linalg.norm(w)
        if w_norm > 10.0:
            w = w / w_norm * 10.0
    
    # Store for analysis
    input_audio.extend(x)
    output_audio.extend(y_out)
    error_audio.extend(e_out)
    
    # Output anti-noise (in real systems, this goes to speaker)
    outdata[:] = y_out.reshape(-1, 1)

def save_audio():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_wav(audio):
        audio = np.array(audio, dtype=np.float32)
        if len(audio) == 0:
            return np.array([], dtype=np.int16)
        max_val = np.abs(audio).max()
        if max_val > 1e-6:
            audio = audio / max_val * 0.95
        return (audio * 32767).astype(np.int16)
    
    if len(input_audio) > 0:
        # Create output directory if it doesn't exist
        # Output saved to current working directory
        wavfile.write(f"input_{timestamp}.wav", fs, to_wav(input_audio))
        wavfile.write(f"antinoise_{timestamp}.wav", fs, to_wav(output_audio))
        wavfile.write(f"cleaned_{timestamp}.wav", fs, to_wav(error_audio))
        
        duration = len(input_audio) / fs
        input_rms = np.sqrt(np.mean(np.array(input_audio)**2))
        error_rms = np.sqrt(np.mean(np.array(error_audio)**2))
        
        if error_rms > 0:
            reduction_db = 10 * np.log10((input_rms**2 + 1e-10) / (error_rms**2 + 1e-10))
        else:
            reduction_db = 0
            
        print(f"\nSaved {duration:.1f}s recording")
        print(f"Noise Reduction: {reduction_db:.2f} dB")
        print(f"Files: input_{timestamp}.wav, antinoise_{timestamp}.wav, cleaned_{timestamp}.wav")

try:
    with sd.Stream(channels=1, samplerate=fs, blocksize=frame_len,
                   dtype='float32', callback=callback):
        print("\nANC ACTIVE - canceling noise...")
        while recording:
            sd.sleep(1000)
            
except KeyboardInterrupt:
    print("\n\nStopping...")
finally:
    recording = False
    save_audio()
    print("\nDone.\n")

