"""
Fixed-alpha FFT preprocessing pipeline 

"""

import numpy as np
from scipy.signal import resample as sci_resample

# ── Configuration ──────────────────────────────────────────────────────────
N          = 2500       
REF_RPM    = 1000.0     
FFT_LO     = 1          
FFT_HI     = 251        
N_BINS     = 250        
LF_BINS    = 3          
ALPHA      = 11.3       

_hann = np.hanning(N)   


def extract_spectrum(raw, actual_rpm):
    """
    Preprocess one raw vibration sample into a speed-invariant log-magnitude spectrum.

    Parameters
    ----------
    raw : np.ndarray, shape (2500, 6)
        Raw accelerometer data. Columns: [Ch1_X, Ch1_Y, Ch1_Z, Ch2_X, Ch2_Y, Ch2_Z].
        The original Time column must already be removed.
    actual_rpm : float
        Measured shaft speed in RPM for this sample.

    Returns
    -------
    spectrum : np.ndarray, shape (250, 6), dtype float32
        Speed-invariant log-magnitude spectrum ready for model input.
        Flatten to (1500,) for ML models or neural network input layers.
    """

    # ── Step 1: Angular resampling ─────────────────────────────────────────
    n_resampled = int(round(N * actual_rpm / REF_RPM))
    resampled = sci_resample(raw.astype(np.float64), n_resampled, axis=0)[:N, :]

    # ── Step 2: Hann window ────────────────────────────────────────────────
    windowed = resampled * _hann[:, np.newaxis]

    # ── Step 3: FFT magnitude ──────────────────────────────────────────────
    magnitude = np.abs(np.fft.rfft(windowed, axis=0))

    # ── Step 4: Log-compression (log1p) ───────────────────────────────────
    L = np.log1p(magnitude[FFT_LO:FFT_HI, :]).astype(np.float32)

    # ── Step 5: Fixed-alpha speed correction ──────────────────────────────
    log_r = np.log(actual_rpm / REF_RPM)
    L -= 2.0 * log_r
    L[:LF_BINS, :] -= (ALPHA - 2.0) * log_r

    return L  


if __name__ == '__main__':
    raw = np.random.randn(N, 6).astype(np.float64)
    spec = extract_spectrum(raw, actual_rpm=1000.0)
    print(f"Output shape : {spec.shape}")  
    print(f"Flattened    : {spec.flatten().shape}") 
    print(f"dtype        : {spec.dtype}")
    print(f"Value range  : [{spec.min():.3f}, {spec.max():.3f}]")
