#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:59:04 2025

@author: laurajorgensen
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Parametre ---
M = 100
f_s = 44100  # Normeret samplingsfrekvens, dvs. pi svarer til 1
f_c_lp = 3400  # Lavpas cutoff (normeret)
f_c1_bp = 200  # Båndpas laveste cutoff
f_c2_bp = 3500  # Båndpas højeste cutoff



n = np.arange(M + 1)
N_fft = 4096
omega = np.linspace(0, np.pi, N_fft//2)

w = 0.5 * (1 - np.cos(2 * np.pi * n / M))  # Hann-vindue

# --- Filterfunktioner ---
def lowpass_filter(f_c, f_s):
    omega_c = 2*np.pi*f_c/f_s
    h_d = np.zeros(M + 1)
    
    for n in range(M + 1):
        if n == M/2:
            h_d[n] = omega_c / np.pi  
        else:
            h_d[n] = np.sin(omega_c * (n - M/2)) / (np.pi * (n - M/2))    
    
    h = h_d * w
    return h

def bandpass_filter(f_c_1, f_c_2, f_s):
    omega_c_1 = 2*np.pi*f_c_1/f_s
    omega_c_2 = 2*np.pi*f_c_2/f_s
    h_d = np.zeros(M + 1)
    for n in range(M + 1):
        if n == M/2:
            h_d[n] = (omega_c_2 - omega_c_1) / np.pi  
        else: 
            h_d[n] = (np.sin(omega_c_2 * (n - M/2)) - np.sin(omega_c_1 * (n - M/2))) / (np.pi * (n - M/2))    
    
    n = np.arange(M + 1)
    h = h_d * w
    return h

# --- FFT funktion ---
def compute_dtft(signal):
    W_fft = np.fft.fft(signal, N_fft)
    W_fft /= np.max(np.abs(W_fft))
    return 20 * np.log10(np.abs(W_fft[:N_fft//2]))

# --- FFT af vinduer ---
Hann_dtft = compute_dtft(w)

# --- Design af filtre ---
h_lp = lowpass_filter(f_c_lp, f_s)
h_bp = bandpass_filter(f_c1_bp, f_c2_bp, f_s)

# --- FFT af filtre ---
LP_dtft = compute_dtft(h_lp)
BP_dtft = compute_dtft(h_bp)

# --- Plot: Lavpas filter ---
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(omega, LP_dtft, label='Lowpass Filter')
plt.plot(2*np.pi*3800/f_s,-30, '+')
plt.plot(2*np.pi*3250/f_s,-0.5, '+')
#plt.title('Lowpass Filter Amplituderespons')
plt.xlabel('Frequency [rad]')
plt.ylabel('Magnitude [dB]')
plt.grid(True, which='both', linewidth=0.5)
plt.ylim(-100, 5)
plt.xlim(0, np.pi)
plt.savefig(f'lowpass_response_M_{M}.pdf')
plt.show()

# --- Plot: Båndpas filter ---
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(omega, BP_dtft, label='Bandpass Filter')
plt.plot(2*np.pi*150/f_s,-20, '+')
plt.plot(2*np.pi*500/f_s,-0.5, '+')
plt.plot(2*np.pi*3200/f_s,-0.5, '+')
plt.plot(2*np.pi*3800/f_s,-40, '+')
plt.ylim(-45, 5)
plt.xlim(0, 0.6)
#plt.title('Bandpass Filter Amplituderespons')
plt.xlabel('Frequency [rad]')
plt.ylabel('Magnitude [dB]')
plt.grid(True, which='both', linewidth=0.5)
# plt.ylim(-100, 5)
# plt.xlim(0, np.pi)
plt.savefig(f'Nbandpass_response_M_{M}.pdf')
plt.show()
