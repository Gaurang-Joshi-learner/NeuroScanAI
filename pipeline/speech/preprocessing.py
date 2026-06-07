"""
pipeline/speech/preprocessing.py
----------------------------------
Loads KaraOne dataset (.mat files), preprocesses EEG for imagined speech.

KaraOne dataset:
  - 14 subjects
  - 11 prompts: /iy/, /uw/, /piy/, /tiy/, /diy/, /m/, /n/, 
                'pat', 'pot', 'knew', 'gnaw'
  - 64 EEG channels, 1000 Hz sampling rate
  - Pre-epoched: each trial is already segmented
  - Files: MM05.mat, MM08.mat, ... per subject
"""

import numpy as np
from scipy import io, signal
from pathlib import Path

SFREQ_KARAONE  = 1000   # Hz
TARGET_SFREQ   = 256    # downsample to match CHB-MIT pipeline
BANDPASS_LOW   = 1.0
BANDPASS_HIGH  = 40.0

PROMPTS = ['/iy/', '/uw/', '/piy/', '/tiy/', '/diy/',
           '/m/', '/n/', 'pat', 'pot', 'knew', 'gnaw']
PROMPT_TO_IDX = {p: i for i, p in enumerate(PROMPTS)}


def bandpass_filter(data: np.ndarray, sfreq: float,
                    low: float, high: float) -> np.ndarray:
    nyq = sfreq / 2
    b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def downsample(data: np.ndarray, orig_sfreq: float,
               target_sfreq: float) -> np.ndarray:
    factor = int(orig_sfreq / target_sfreq)
    return data[..., ::factor]


def load_karaone_subject(mat_path: str) -> tuple:
    """
    Load a single KaraOne .mat file.
    Returns:
        epochs: (n_trials, n_channels, n_samples)
        labels: (n_trials,) integer class indices
        meta:   dict
    """
    mat = io.loadmat(mat_path, squeeze_me=True)

    # KaraOne stores data in 'eeg' field, labels in 'prompts'
    # Structure varies by version — handle both
    if 'eeg' in mat:
        eeg_data = mat['eeg']        # (n_trials, n_channels, n_samples) or struct
    elif 'EEG' in mat:
        eeg_data = mat['EEG']
    else:
        raise KeyError(f'Cannot find EEG data in {mat_path}. Keys: {list(mat.keys())}')

    prompts = mat.get('prompts', mat.get('labels', None))
    if prompts is None:
        raise KeyError('Cannot find prompt labels in mat file')

    # Convert string prompts to integer labels
    if hasattr(prompts, '__iter__') and not isinstance(prompts, str):
        labels = np.array([PROMPT_TO_IDX.get(str(p).strip(), -1) for p in prompts])
    else:
        labels = np.array([PROMPT_TO_IDX.get(str(prompts).strip(), -1)])

    # Filter out unknown prompts
    valid = labels >= 0
    eeg_data = eeg_data[valid]
    labels = labels[valid]

    # Preprocess each trial
    processed = []
    for trial in eeg_data:
        # trial: (n_channels, n_samples)
        filtered = bandpass_filter(trial, SFREQ_KARAONE, BANDPASS_LOW, BANDPASS_HIGH)
        downsampled = downsample(filtered, SFREQ_KARAONE, TARGET_SFREQ)
        processed.append(downsampled)

    epochs = np.array(processed, dtype=np.float32)

    meta = {
        'subject': Path(mat_path).stem,
        'n_trials': len(labels),
        'n_channels': epochs.shape[1],
        'n_samples': epochs.shape[2],
        'sfreq': TARGET_SFREQ,
        'prompts': PROMPTS,
        'class_counts': {PROMPTS[i]: int((labels==i).sum()) for i in range(len(PROMPTS))},
    }

    return epochs, labels, meta


def load_karaone_dataset(data_dir: str) -> tuple:
    """
    Load all subjects from KaraOne directory.
    Returns concatenated epochs, labels, and meta.
    """
    data_dir = Path(data_dir)
    mat_files = sorted(data_dir.glob('*.mat'))

    if not mat_files:
        raise FileNotFoundError(
            f'No .mat files found in {data_dir}.\n'
            f'Download KaraOne from http://www.cs.toronto.edu/~complingweb/data/karaOne/'
        )

    all_epochs, all_labels = [], []
    meta = None

    for mat_path in mat_files:
        print(f'  Loading {mat_path.name}...', end=' ')
        try:
            epochs, labels, m = load_karaone_subject(str(mat_path))
            all_epochs.append(epochs)
            all_labels.append(labels)
            if meta is None:
                meta = m
            print(f'{len(labels)} trials')
        except Exception as e:
            print(f'FAILED: {e}')

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f'\nTotal: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples')
    print(f'Classes: {len(PROMPTS)} prompts')

    meta['total_trials'] = len(y)
    return X, y, meta
