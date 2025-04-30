import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram

# --- Parametre ---
M = 300
f_c1_bp = 300
f_c2_bp = 3400

# --- Hann-vindue og indeks ---
n = np.arange(M + 1)
w = 0.5 * (1 - np.cos(2 * np.pi * n / M))

# --- Bandpass-filterdesign ---
def bandpass_filter(f_c_1, f_c_2, f_s):
    omega_c_1 = 2 * np.pi * f_c_1 / f_s
    omega_c_2 = 2 * np.pi * f_c_2 / f_s
    h_d = np.zeros(M + 1)
    for n in range(M + 1):
        if n == M / 2:
            h_d[n] = (omega_c_2 - omega_c_1) / np.pi
        else:
            h_d[n] = (np.sin(omega_c_2 * (n - M/2)) - np.sin(omega_c_1 * (n - M/2))) / (np.pi * (n - M/2))
    return h_d * w

# --- FFT-funktion ---
def compute_fft(signal, fs):
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    spectrum = np.abs(fft(signal))
    return freqs[:N//2], spectrum[:N//2]

# --- Læs lydfil ---
sample_freq, audio_data = wavfile.read('sentences/DT2_Danish_Speech_L1_S1.wav')
f_s = sample_freq
t = np.linspace(0, len(audio_data) / f_s, len(audio_data), endpoint=False)

    
# --- Design filter og filtrér ---?
h_bp = bandpass_filter(f_c1_bp, f_c2_bp, f_s)
filtered_audio_data = np.convolve(audio_data, h_bp, mode='same')  # Full convolution

# --- Plot ---
plt.figure(figsize=(10, 3), dpi=300)
plt.plot(t, audio_data, label="x(t)")
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True, which='both', linewidth=0.5)
plt.ylim([-7000, 7000]) 
plt.xlim([0, len(audio_data) / f_s])  
plt.tight_layout()
plt.savefig('og_signal.pdf')
plt.show()

# --- Plot ---
plt.figure(figsize=(10, 3), dpi=300)
plt.plot(t, filtered_audio_data, label="x(t)")
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True, which='both', linewidth=0.5)
plt.ylim([-7000, 7000])  # Justér hvis nødvendigt
plt.xlim([0, len(audio_data) / f_s])    # Viser kun første sekund
plt.tight_layout()
plt.savefig('filtered_signal.pdf')
plt.show()

def plot_spectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs, nperseg=256)
    plt.figure(figsize=(10, 4), dpi=400)
    plt.pcolormesh(t, f, 20 * np.log10(Sxx), shading='nearest', cmap='viridis',vmin=-100, vmax=50)
    plt.colorbar() #label='Amplitude [dB]'
    plt.ylabel('Frequency [Hz]',  fontsize=14)
    plt.xlabel('Time [s]', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

plot_spectrogram(audio_data, sample_freq, 'audio_data')
plot_spectrogram(filtered_audio_data, sample_freq, 'filtered_audio_data')
