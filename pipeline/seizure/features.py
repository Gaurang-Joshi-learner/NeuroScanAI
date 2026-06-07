"""
pipeline/seizure/features.py
-----------------------------
Extracts clinically validated EEG features for seizure detection.

Feature set (per channel):
  1. Band power: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-40)
  2. Hjorth parameters: activity, mobility, complexity
  3. Statistical: mean, variance, skewness, kurtosis, zero-crossing rate
  4. Spectral entropy
  5. Relative band power ratios (theta/alpha, beta/alpha — known seizure markers)

Total features per epoch: n_channels × 16 features
For 23 channels: 368 features
"""

import numpy as np
from scipy import signal, stats


# ── Band definitions (Hz) ────────────────────────────────
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 40),
}


def bandpower(data: np.ndarray, sfreq: float, low: float, high: float) -> float:
    """Compute average power in a frequency band using Welch's method."""
    freqs, psd = signal.welch(data, sfreq, nperseg=min(256, len(data)))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))


def hjorth_parameters(data: np.ndarray) -> tuple:
    """
    Hjorth activity, mobility, complexity.
    Standard features for EEG pathology detection.
    """
    activity   = np.var(data)
    diff1      = np.diff(data)
    diff2      = np.diff(diff1)
    mobility   = np.sqrt(np.var(diff1) / (activity + 1e-10))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity


def spectral_entropy(data: np.ndarray, sfreq: float) -> float:
    """Spectral entropy — lower during seizure (more ordered signal)."""
    freqs, psd = signal.welch(data, sfreq, nperseg=min(256, len(data)))
    psd_norm = psd / (psd.sum() + 1e-10)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))


def zero_crossing_rate(data: np.ndarray) -> float:
    """Rate of sign changes — higher during high-frequency seizure activity."""
    return float(np.mean(np.diff(np.sign(data)) != 0))


def extract_channel_features(channel_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extract all features for a single channel.
    Returns 1D array of 16 features.
    """
    features = []

    # 1. Band powers (5 features)
    band_powers = {}
    total_power = 0
    for band, (low, high) in BANDS.items():
        bp = bandpower(channel_data, sfreq, low, high)
        band_powers[band] = bp
        total_power += bp
        features.append(bp)

    # 2. Relative band power ratios (2 features — known seizure markers)
    theta_alpha = band_powers['theta'] / (band_powers['alpha'] + 1e-10)
    beta_alpha  = band_powers['beta']  / (band_powers['alpha'] + 1e-10)
    features.extend([theta_alpha, beta_alpha])

    # 3. Hjorth parameters (3 features)
    activity, mobility, complexity = hjorth_parameters(channel_data)
    features.extend([activity, mobility, complexity])

    # 4. Statistical features (4 features)
    features.append(float(np.mean(np.abs(channel_data))))   # mean absolute value
    features.append(float(np.var(channel_data)))            # variance
    features.append(float(stats.skew(channel_data)))        # skewness
    features.append(float(stats.kurtosis(channel_data)))    # kurtosis

    # 5. Spectral entropy (1 feature)
    features.append(spectral_entropy(channel_data, sfreq))

    # 6. Zero crossing rate (1 feature)
    features.append(zero_crossing_rate(channel_data))

    return np.array(features, dtype=np.float32)


def extract_features(epochs: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Extract features from all epochs.

    Args:
        epochs: (N, C, T) — N epochs, C channels, T time samples
        sfreq:  sampling frequency

    Returns:
        features: (N, C*16) feature matrix
    """
    N, C, T = epochs.shape
    n_features_per_channel = 16
    features = np.zeros((N, C * n_features_per_channel), dtype=np.float32)

    print(f'Extracting features: {N} epochs × {C} channels × {n_features_per_channel} features...')

    for i in range(N):
        if i % 100 == 0:
            print(f'  Epoch {i}/{N}')
        for c in range(C):
            start = c * n_features_per_channel
            end   = start + n_features_per_channel
            features[i, start:end] = extract_channel_features(epochs[i, c], sfreq)

    # Replace NaN/Inf with 0 (can occur from near-flat channels)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print(f'Feature matrix: {features.shape}')
    return features
