import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows

# Define parameters
fs = 5000  # Sampling frequency (Hz)
T = 1      # Signal duration (seconds)
t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector

# Define signal components
signal = np.cos(np.pi * 100 * t) + np.cos(2 * np.pi * 450 * t)

# Add impulse (approximated by a large value at the closest sample)
impulse_times = [0.20, 0.5, 0.8]
impulse_signal = np.zeros_like(t)
for imp in impulse_times:
    idx = np.argmin(np.abs(t - imp))  # Find closest sample
    impulse_signal[idx] = 6  # Approximate δ(t) with a discrete spike

# Combine signals
x_t = signal + impulse_signal

# Plot the time-domain signal
plt.figure(figsize=(10, 2), dpi=150)
plt.plot(t, x_t, label="x(t)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [Hz]")
plt.title("Time Domain Signal")
plt.ylim([-3, 10])  # Set y-axis range
plt.tight_layout()
plt.show()

# Spectrogram parameters
window_lengths = [100, 600]
titles = [
    "Spectrogram of x(t) with window length L=100",
    "Spectrogram of x(t) with window length L=600"
]
colormap = "viridis"

# Generate and plot each spectrogram in its own figure
for L, title in zip(window_lengths, titles):
    f, t_spec, Sxx = spectrogram(
        x_t, fs, window=windows.hann(L), nperseg=L, noverlap=L//2, nfft=1024
    )

    plt.figure(figsize=(5, 3), dpi=150)
    plt.pcolormesh(t_spec, f, 20 * np.log10(Sxx), shading='auto', cmap=colormap)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar()
    plt.ylim([0, 1000])
    plt.title(title)
    plt.tight_layout()
    plt.show()
