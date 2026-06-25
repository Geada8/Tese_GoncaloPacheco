"""
Fixed-alpha FFT preprocessing pipeline for vibration fault detection.
Converts a raw 2500x6 accelerometer sample into a speed-invariant
log-magnitude spectrum of shape (250, 6) = 1500 features when flattened.

Usage:
    spectrum = extract_spectrum(raw_array, actual_rpm)
"""

import numpy as np
from scipy.signal import resample as sci_resample

# ── Configuration ──────────────────────────────────────────────────────────
N          = 2500       # samples per window (1 shaft revolution at 25 kHz / ~1000 RPM)
REF_RPM    = 1000.0     # reference RPM used during training
FFT_LO     = 1          # first bin kept (skip DC at bin 0)
FFT_HI     = 251        # last bin kept exclusive (bins 1-250 = 10-2490 Hz)
N_BINS     = 250        # number of frequency bins in output
LF_BINS    = 3          # bins covering the 1x shaft-order region (bins 1-3, ~10-30 Hz)
ALPHA      = 11.3       # fixed exponent for structural resonance correction

_hann = np.hanning(N)   # precomputed Hann window


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
    # Resample so that exactly N samples correspond to one shaft revolution,
    # regardless of the actual shaft speed. This aligns the spectral bins
    # across different operating speeds before the FFT.
    n_resampled = int(round(N * actual_rpm / REF_RPM))
    resampled = sci_resample(raw.astype(np.float64), n_resampled, axis=0)[:N, :]

    # ── Step 2: Hann window ────────────────────────────────────────────────
    # Reduces spectral leakage caused by the finite-length signal window.
    windowed = resampled * _hann[:, np.newaxis]

    # ── Step 3: FFT magnitude ──────────────────────────────────────────────
    # Compute one-sided magnitude spectrum. Keep only bins FFT_LO:FFT_HI
    # (bins 1-250), discarding DC and high-frequency content above 2490 Hz.
    magnitude = np.abs(np.fft.rfft(windowed, axis=0))

    # ── Step 4: Log-compression (log1p) ───────────────────────────────────
    # Compresses the dynamic range of the spectrum, making the model less
    # sensitive to absolute amplitude differences between samples.
    L = np.log1p(magnitude[FFT_LO:FFT_HI, :]).astype(np.float32)

    # ── Step 5: Fixed-alpha speed correction ──────────────────────────────
    # Removes the speed-dependent amplitude scaling so that spectra recorded
    # at different RPMs are directly comparable.
    #
    # Physics: vibration amplitude scales as omega^alpha where alpha ~ 11.3
    # for structural resonance and ~2 for broadband forcing.
    #
    # Broadband correction (all 250 bins):
    #   L -= 2.0 * log(rpm / ref_rpm)
    #
    # Resonance correction (first 3 bins, 1x shaft-order region):
    #   L -= (alpha - 2.0) * log(rpm / ref_rpm)
    log_r = np.log(actual_rpm / REF_RPM)
    L -= 2.0 * log_r
    L[:LF_BINS, :] -= (ALPHA - 2.0) * log_r

    return L  # shape: (250, 6)


if __name__ == '__main__':
    # Quick sanity check
    raw = np.random.randn(N, 6).astype(np.float64)
    spec = extract_spectrum(raw, actual_rpm=1000.0)
    print(f"Output shape : {spec.shape}")   # (250, 6)
    print(f"Flattened    : {spec.flatten().shape}")  # (1500,)
    print(f"dtype        : {spec.dtype}")
    print(f"Value range  : [{spec.min():.3f}, {spec.max():.3f}]")
