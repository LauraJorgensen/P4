#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:59:04 2025

@author: laurajorgensen
"""

import numpy as np
import matplotlib.pyplot as plt

# Antal samples og akse
M = 50
n = np.arange(M + 1)

# Vinduesfunktioner
hann = np.hanning(M + 1)
bartlett = np.bartlett(M + 1)
rectangular = np.ones(M + 1)

# --- Plot: Vinduesfunktioner ---
plt.figure(figsize=(6, 4), dpi=300)
plt.xlim(0, M)
plt.ylim(0, 1.05)
plt.plot(n, rectangular, color='k', label='Rectangular')
plt.plot(n, hann, '--', label='Hann', linewidth=1.5)
plt.plot(n, bartlett, '--', label='Bartlett', linewidth=1.5)
plt.xticks([0, M/2, M], ['0', r'$M/2$', 'M'])
plt.yticks(np.linspace(0, 1, 6))
plt.xlabel(r'$n$', fontsize=14)
plt.ylabel(r'$\omega[n]$', fontsize=14)
plt.grid(True, which='both', linewidth=0.5)
plt.savefig('filter_design_techniques.pdf')
plt.show()

# --- Plot: DTFT (FFT af vinduer) ---
N_fft = 4096
w = np.linspace(0, np.pi, N_fft//2)

# FFT 
def compute_dtft(window):
    W_fft = np.fft.fft(window, N_fft)
    W_fft /= np.max(np.abs(W_fft))
    return 20 * np.log10(np.abs(W_fft[:N_fft//2]))

Hann_dtft = compute_dtft(hann)
Bartlett_dtft = compute_dtft(bartlett)
Rectangular_dtft = compute_dtft(rectangular)

# Plot DTFT
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(w, Rectangular_dtft, color='k', label='Rectangular', linewidth=1)
plt.plot(w, Hann_dtft, '--', label='Hann', linewidth=1.5)
plt.plot(w, Bartlett_dtft, '--', label='Bartlett', linewidth=1.5)
plt.xlim(0, np.pi)
plt.ylim(-100, 5)
plt.xticks(np.linspace(0, np.pi, 6),
           [r'$0$', r'$0.2\pi$', r'$0.4\pi$', r'$0.6\pi$', r'$0.8\pi$', r'$\pi$'])
plt.xlabel('$\omega$ [rad]', fontsize=14)
plt.ylabel('Magnitude [dB]', fontsize=14)
plt.grid(True, which='both', linewidth=0.5)
plt.savefig('window_dtft_normalized.pdf')
plt.show()
