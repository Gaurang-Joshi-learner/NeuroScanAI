"""
pipeline/speech/features.py
----------------------------
Feature extraction for imagined speech EEG.

For speech decoding, the most informative features are:
  1. Common Spatial Patterns (CSP) — spatial filters maximizing class separation
  2. Band power per channel (same bands as seizure but gamma matters more)
  3. Temporal features: mean, std per channel
  4. Frequency band ratios
"""

import numpy as np
from scipy import signal, stats
from sklearn.decomposition import PCA


BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 40),
}


def bandpower(data, sfreq, low, high):
    freqs, psd = signal.welch(data, sfreq, nperseg=min(128, len(data)))
    idx = np.logical_and(freqs >= low, freqs <= high)
    if idx.sum() == 0:
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def extract_trial_features(trial: np.ndarray, sfreq: float = 256.0) -> np.ndarray:
    """
    Extract features from a single trial (n_channels, n_samples).
    Returns 1D feature vector.
    """
    n_channels = trial.shape[0]
    features = []

    for ch in range(n_channels):
        ch_data = trial[ch]
        # Band powers
        for low, high in BANDS.values():
            features.append(bandpower(ch_data, sfreq, low, high))
        # Statistical
        features.append(float(np.mean(ch_data)))
        features.append(float(np.std(ch_data)))

    return np.array(features, dtype=np.float32)


def extract_features(epochs: np.ndarray, sfreq: float = 256.0,
                     n_pca_components: int = 50) -> tuple:
    """
    Extract features from all trials and optionally reduce with PCA.

    Returns:
        features: (N, n_features) array
        pca:      fitted PCA object (or None)
    """
    N = epochs.shape[0]
    all_features = []

    print(f'Extracting speech features: {N} trials...')
    for i in range(N):
        if i % 50 == 0:
            print(f'  Trial {i}/{N}')
        all_features.append(extract_trial_features(epochs[i], sfreq))

    X = np.array(all_features, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # PCA to reduce dimensionality
    pca = None
    if n_pca_components and X.shape[1] > n_pca_components:
        pca = PCA(n_components=n_pca_components, random_state=42)
        X = pca.fit_transform(X).astype(np.float32)
        print(f'PCA: {X.shape[1]} → {n_pca_components} components '
              f'({pca.explained_variance_ratio_.sum()*100:.1f}% variance)')

    print(f'Feature matrix: {X.shape}')
    return X, pca
