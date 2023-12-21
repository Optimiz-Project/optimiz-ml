import librosa
import numpy as np
from scipy.signal import medfilt

def median_filter(signal, kernel_size=3):
    return medfilt(signal, kernel_size)

# Fungsi untuk mengurangi noise dengan spektral subtraksi
def spectral_subtraction(signal, noise, alpha=1):
    signal_energy = np.sum(np.abs(signal)**2)
    noise_energy = np.sum(np.abs(noise)**2)
    reduction_factor = np.maximum(1.0 - alpha * (noise_energy / signal_energy), 0)
    denoised_signal = signal * reduction_factor
    return denoised_signal
