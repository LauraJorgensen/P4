# --- Importer nødvendige pakker ---
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
from datetime import datetime

# --- parametre

filename = 'sentences/DT2_Danish_Speech_L6_S1.wav'
background_noise_filename = 'Noise_fredagsbar.wav'

N = 2**11
M_LP = 270
M_BP = 300
f_c_1 = 300
f_c_2 = 3400

timestamp = datetime.now().strftime("%H%M%S")
base_name = os.path.splitext(os.path.basename(filename))[0]
OUTPUT_DIR = f"F_{N}"
#OUTPUT_DIR = f"ResultsF_{base_name}_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Indlæs lydfil ---
def load_data(filename):
    sample_freq, audio_data = wavfile.read(filename)
    return sample_freq, audio_data

# --- Konverter signal vektor til wavfile.
def convert_to_wav(array, output_filename, audio_data, sample_freq):
    array = np.asarray(array, dtype=audio_data.dtype) 
    wavfile.write(os.path.join(OUTPUT_DIR, output_filename), sample_freq, array)


# --- Zeropadding på signalet ---
def reshape(array):
    remainder = len(array) % N
    if remainder == 0:
        pad_length = 0  
    else: 
        pad_length = N - remainder
    audio_data = np.pad(array, (0, pad_length))

    audio_data_matrix = audio_data.reshape(-1, N).T
    return audio_data_matrix, pad_length


# --- Frequency scrambling af signalet ---
def scramble(signal):
    spectrum = np.fft.fft(signal, axis=0)
    
    np.random.seed(42)  
    permutation_idx = np.random.permutation(N)
    P = np.eye(N)[permutation_idx]
    scrampled_spectrum = P @ spectrum

    scrampled_signal = np.fft.ifft(scrampled_spectrum, axis=0).flatten('F').real
    
    return scrampled_signal, P


# --- Inversion scramble til eksempel ---
def reverse_scramble(signal_matrix):
    spectrum = np.fft.fft(signal_matrix, axis=0)
    reversed_spectrum = np.fft.fftshift(spectrum, axes=0)
    signal_out = np.fft.ifft(reversed_spectrum, axis=0).real
    return signal_out.flatten('F')



# --- Frequency descrambling af signalet ---
def descramble(scrampled_signal, P, pad_length):
    scrampled_signal_matrix = scrampled_signal.reshape(-1, N).T
    spectrum_scrampled_signal = np.fft.fft(scrampled_signal_matrix, axis=0)

    inverse_permutation = P.T
    spectrum_descrampled_signal = inverse_permutation @ spectrum_scrampled_signal

    descrambled_signal = np.fft.ifft(spectrum_descrampled_signal, axis=0).flatten('F').real

    descrambled_signal  = descrambled_signal[:-pad_length]
    
    return descrambled_signal



# --- Tilføjelse af baggrundsstøj på lydfiler ---
def add_background_noise(signal, noise_filename):
    sample_freq, noise = load_data(noise_filename)
    
    noise = noise[:len(signal)]

    signal_noisy = signal + noise*0.05
    signal_noisy = np.clip(signal_noisy, -32768, 32767).astype(np.int16)

    return signal_noisy


# --- Lavpas filter udledt i kapitel 9 ---
def lowpass_filter(M, f_c, f_s):
    omega_c = 2*np.pi*f_c/f_s
    h_d = np.zeros(M + 1)
    
    for n in range(M + 1):
        if n == M/2:
            h_d[n] = omega_c / np.pi  
        else:
            h_d[n] = np.sin(omega_c * (n - M/2)) / (np.pi * (n - M/2))    
    
    n = np.arange(M + 1)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / M))
    h = h_d * w
    return h


# --- Båndpas filter udledt i kapitel 9 ---
def bandpass_filter(M, f_c_1, f_c_2, f_s):
    omega_c_1 = 2*np.pi*f_c_1/f_s
    omega_c_2 = 2*np.pi*f_c_2/f_s
    h_d = np.zeros(M + 1)
    for n in range(M + 1):
        if n == M/2:
            h_d[n] = (omega_c_2-omega_c_1) / np.pi  
        else: 
            h_d[n] = (np.sin(omega_c_2 * (n - M/2))-np.sin(omega_c_1 * (n - M/2)) )/ (np.pi * (n - M/2))    
    
    n = np.arange(M + 1)
    w = 0.5 * (1 - np.cos(2 * np.pi * n / M))
    h = h_d * w
    return h

# --- Tilføjelse af hvidstøj på lydfiler ---
def add_gaussian_noise(signal, snr_db):
    rms = np.sqrt(np.mean(signal**2))
    
    rms_noise = rms / (10**(snr_db / 20))
    
    noise = np.random.normal(0, 1, size=signal.shape)
    noise = noise / np.sqrt(np.mean(noise**2))  
    noise = noise * rms_noise               
    
    noisy_signal = signal + noise
    
    return noisy_signal

# --- Plot af spectogrammer --- 
def plot_spectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs, nperseg=N)
    plt.figure(figsize=(10, 4), dpi=400)
    plt.pcolormesh(t, f, 20 * np.log10(Sxx), shading='nearest', cmap='viridis',vmin=-100, vmax=50)
    plt.colorbar() #label='Amplitude [dB]'
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.xlabel('Time [s]',fontsize=14)
    #plt.title(title)
    plt.tight_layout()
    safe_title = title.replace(" ", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe_title}.png"))
    plt.show()
    

# --- Sammensætning af overstående funktioner til simulering af hele scrambler/descrambler systemet med tilføjelse af støj med filter ---
def main_noisy(tal):
    sample_freq, audio_data = load_data(filename)
    audio_data_noisy = add_background_noise(audio_data, background_noise_filename)

    h_BP = bandpass_filter(M_BP, f_c_1, f_c_2, sample_freq)
    h_LP = lowpass_filter(M_LP, f_c_2, sample_freq)
    
    audio_data_filtered = np.convolve(audio_data_noisy, h_BP)
    
    audio_data_matrix, pad_length = reshape(audio_data_filtered)
    scrampled_signal, P = scramble(audio_data_matrix)

    scrampled_signal_noisy = add_gaussian_noise(scrampled_signal, 20) 
    convert_to_wav(scrampled_signal_noisy, f'{tal}_scrampled_signal.wav', audio_data, sample_freq)

    descrambled_signal_noisy = descramble(scrampled_signal_noisy, P, pad_length)
    # convert_to_wav(descrambled_signal_noisy, f'{tal}_descrambled_signal_noisy.wav', audio_data, sample_freq)
    
    descrambled_signal_filteret = np.convolve(descrambled_signal_noisy, h_LP)
    convert_to_wav(descrambled_signal_filteret, f'{tal}_descrambled_signal_filter.wav', audio_data, sample_freq)

    # plot_spectrogram(audio_data, sample_freq, "Original signal")
    plot_spectrogram(audio_data_noisy, sample_freq, "Original signal with noise")
    plot_spectrogram(audio_data_filtered, sample_freq, "Original signal filtered")
    plot_spectrogram(scrampled_signal_noisy, sample_freq, "Scrambled signal with noise")
    # plot_spectrogram(descrambled_signal_noisy, sample_freq, "Descrambled signal with noise")
    plot_spectrogram(descrambled_signal_filteret, sample_freq, "Descrambled signal filtered")
    
    
# --- Sammensætning af overstående funktioner til simulering af hele scrambler/descrambler systemet uden tilføjet støj og uden filter---
def main_clean(tal):
    sample_freq, audio_data = load_data(filename)
    
    audio_data_matrix, pad_length = reshape(audio_data)
    scrampled_signal, P = scramble(audio_data_matrix)
    #inverse_signal = inverse(audio_data_matrix)
    #rev = reverse_scramble(audio_data_matrix)
    convert_to_wav(scrampled_signal, f'{tal}_scrampled_signal.wav', audio_data, sample_freq)
    
    descrambled_signal = descramble(scrampled_signal, P, pad_length)
    convert_to_wav(descrambled_signal, f'{tal}_descrambled_signal.wav', audio_data, sample_freq)
    
    # plot_spectrogram(audio_data, sample_freq, "Original signal")
    # plot_spectrogram(scrampled_signal, sample_freq, "Scrambled signal")
    # plot_spectrogram(rev, sample_freq, "Inverse signal")
    # plot_spectrogram(descrambled_signal, sample_freq, "Descrambled signal filtered")

# --- Sammensætning af overstående funktioner til simulering af hele scrambler/descrambler systemet uden tilføjet støj med filter---
def main_filter(tal):
    sample_freq, audio_data = load_data(filename)

    h_BP = bandpass_filter(M_BP, f_c_1, f_c_2, sample_freq)
    h_LP = lowpass_filter(M_LP, f_c_2, sample_freq)
    
    audio_data_filtered = np.convolve(audio_data, h_BP)
    
    audio_data_matrix, pad_length = reshape(audio_data_filtered)
    scrampled_signal, P = scramble(audio_data_matrix)

    convert_to_wav(scrampled_signal, f'{tal}_scrampled_signal.wav', audio_data, sample_freq)

    descrambled_signal_noisy = descramble(scrampled_signal, P, pad_length)
    # convert_to_wav(descrambled_signal_noisy, f'{tal}_descrambled_signal_noisy.wav', audio_data, sample_freq)
    
    descrambled_signal_filteret = np.convolve(descrambled_signal_noisy, h_LP)
    convert_to_wav(descrambled_signal_filteret, f'{tal}_descrambled_signal_filter.wav', audio_data, sample_freq)

    
main_filter(3)
#main_noisy(3)