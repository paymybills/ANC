import numpy as np
from scipy.io import wavfile

def run_fxlms(input_file, output_prefix="fxlms", mu=0.001, N=512):
    # Load input
    fs, data = wavfile.read(input_file)
    if data.dtype == np.int16:
        data = data / 32768.0
    
    num_samples = len(data)
    
    # Secondary path estimate S_hat (Unit pulse if unknown)
    S_hat = np.array([1.0]) 
    L_s = len(S_hat)
    
    # Initialize
    w = np.zeros(N)
    x_buf = np.zeros(N)
    xf_buf = np.zeros(N)
    
    y_out = np.zeros(num_samples)
    e_out = np.zeros(num_samples)
    
    for n in range(num_samples):
        x_buf = np.roll(x_buf, 1)
        x_buf[0] = data[n]
        
        y = np.dot(w, x_buf)
        y_out[n] = y
        
        # In this offline version, we assume error is measured after cancellation
        # Simplified: e = d - y (assuming S(z) = 1 for pulse estimate)
        e = data[n] - y
        e_out[n] = e
        
        x_latest_s = np.dot(S_hat, x_buf[:L_s])
        xf_buf = np.roll(xf_buf, 1)
        xf_buf[0] = x_latest_s
        
        w = w + 2 * mu * e * xf_buf
        
    # Save
    def to_int16(audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0: audio = audio / max_val
        return (audio * 32767).astype(np.int16)
        
    wavfile.write(f"{output_prefix}_cleaned.wav", fs, to_int16(e_out))
    print(f"Processed {input_file} -> {output_prefix}_cleaned.wav")

if __name__ == "__main__":
    print("FxLMS Basic Implementation")
    print("Use: import and call run_fxlms(wav_path)")
