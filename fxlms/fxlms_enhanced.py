import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Enhanced Version (Actually "Slow" version for comparison)
# Parameters from QUICK_SUMMARY.md
fs = 16000
N = 256
mu = 0.00005  # VERY SLOW learning rate
duration = 5  # seconds
num_samples = fs * duration

# Simple 3-tap secondary path S(z)
S_path = np.array([0.5, 1.0, 0.5])
S_path = S_path / np.sum(np.abs(S_path))
L_s = len(S_path)
S_hat = S_path.copy()

# Generate tonal noise (1000Hz Sine + low White Noise)
t = np.linspace(0, duration, num_samples)
x = 0.5 * np.sin(2 * np.pi * 1000 * t) + np.random.normal(0, 0.1, num_samples)

# Initialize
w = np.zeros(N)
x_buf = np.zeros(N)
y_buf = np.zeros(L_s)
xf_buf = np.zeros(N)

# Storage
y_out = np.zeros(num_samples)
e_out = np.zeros(num_samples)

print(f"Running Enhanced FxLMS (Slow Mode)...")
print(f"Params: mu={mu}, N={N}, taps={L_s}")

for n in range(num_samples):
    # Reference buffer
    x_buf = np.roll(x_buf, 1)
    x_buf[0] = x[n]
    
    # 1. Anti-noise
    y = np.dot(w, x_buf)
    y_out[n] = y
    
    # 2. Secondary Path Simulation
    y_buf = np.roll(y_buf, 1)
    y_buf[0] = y
    y_s = np.dot(S_path, y_buf)
    
    # 3. Error
    e = x[n] - y_s
    e_out[n] = e
    
    # 4. Filtered-X
    x_latest_s = np.dot(S_hat, x_buf[:L_s])
    xf_buf = np.roll(xf_buf, 1)
    xf_buf[0] = x_latest_s
    
    # 5. Update
    w = w + 2 * mu * e * xf_buf

# Calculate reduction
reduction = 10 * np.log10(np.var(x) / np.var(e_out))
print(f"Done. Noise Reduction: {reduction:.2f} dB")

# Save results
def to_int16(data):
    return (data / np.max(np.abs(data)) * 32767).astype(np.int16)

wavfile.write("enhanced_input.wav", fs, to_int16(x))
wavfile.write("enhanced_output.wav", fs, to_int16(y_out))
wavfile.write("enhanced_error.wav", fs, to_int16(e_out))

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(x[:1000], label='Input Noise')
plt.plot(e_out[:1000], label='Error Signal', alpha=0.7)
plt.title(f"Enhanced FxLMS (mu={mu}, N={N}) - Start of Adaptation")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x[-1000:], label='Input Noise')
plt.plot(e_out[-1000:], label='Error Signal', alpha=0.7)
plt.title("End of Adaptation (5 seconds)")
plt.legend()
plt.tight_layout()
plt.savefig("enhanced_performance.png")
print("Saved summary plot: enhanced_performance.png")
