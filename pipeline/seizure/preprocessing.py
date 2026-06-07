"""
pipeline/seizure/preprocessing.py
-----------------------------------
Loads CHB-MIT .edf files, applies clinical-grade signal preprocessing,
parses seizure annotations from .edf.seizures files, and produces
labelled 4-second epochs ready for feature extraction.

CHB-MIT specifics:
  - 23 EEG channels, 256 Hz sampling rate
  - Seizure times stored in chb01-summary.txt AND .edf.seizures files
  - Files without a .seizures file = entirely non-seizure recording
"""

import os
import re
import numpy as np
import mne
from pathlib import Path

# ── Constants ────────────────────────────────────────────
SFREQ          = 256          # CHB-MIT sampling rate (Hz)
EPOCH_SEC      = 4            # epoch length in seconds
EPOCH_SAMPLES  = SFREQ * EPOCH_SEC   # 1024 samples
OVERLAP        = 0.5          # 50% overlap between epochs
STEP_SAMPLES   = int(EPOCH_SAMPLES * (1 - OVERLAP))

BANDPASS_LOW   = 0.5          # Hz
BANDPASS_HIGH  = 40.0         # Hz
NOTCH_FREQ     = 60.0         # Hz (US powerline)

# Standard 23-channel CHB-MIT montage
CHB_CHANNELS = [
    'FP1-F7','F7-T7','T7-P7','P7-O1',
    'FP1-F3','F3-C3','C3-P3','P3-O1',
    'FP2-F4','F4-C4','C4-P4','P4-O2',
    'FP2-F8','F8-T8','T8-P8','P8-O2',
    'FZ-CZ','CZ-PZ','P7-T7','T7-FT9',
    'FT9-FT10','FT10-T8','T8-P8'
]


def parse_seizure_file(seizure_file: str) -> list:
    """
    Parse a .edf.seizures file.
    Format:
        Number of seizures in file: 2
        Seizure 1 Start Time: 2996 seconds
        Seizure 1 End Time: 3036 seconds
    Returns list of (start_sec, end_sec) tuples.
    """
    seizures = []
    if not os.path.exists(seizure_file):
        return seizures

    with open(seizure_file, 'r') as f:
        content = f.read()

    starts = re.findall(r'Seizure\s+\d+\s+Start\s+Time:\s+(\d+)', content, re.IGNORECASE)
    ends   = re.findall(r'Seizure\s+\d+\s+End\s+Time:\s+(\d+)',   content, re.IGNORECASE)

    for s, e in zip(starts, ends):
        seizures.append((int(s), int(e)))

    return seizures


def parse_summary_file(summary_path: str) -> dict:
    """
    Parse chb01-summary.txt to get seizure times per file.
    Returns dict: {'chb01_03.edf': [(2996, 3036), ...], ...}
    """
    seizure_map = {}
    if not os.path.exists(summary_path):
        return seizure_map

    with open(summary_path, 'r') as f:
        content = f.read()

    # Split by file blocks
    blocks = re.split(r'File Name:', content)[1:]
    for block in blocks:
        lines = block.strip().split('\n')
        filename = lines[0].strip()
        seizures = []
        starts = re.findall(r'Seizure\s+\d*\s*Start\s+Time:\s+(\d+)', block, re.IGNORECASE)
        ends   = re.findall(r'Seizure\s+\d*\s*End\s+Time:\s+(\d+)',   block, re.IGNORECASE)
        for s, e in zip(starts, ends):
            seizures.append((int(s), int(e)))
        seizure_map[filename] = seizures

    return seizure_map


def load_and_preprocess_edf(edf_path: str) -> tuple:
    """
    Load a single EDF file and apply preprocessing pipeline.
    Returns: (raw_mne_object, sfreq)

    Pipeline:
        1. Load with MNE
        2. Pick EEG channels only
        3. Set common average reference
        4. Bandpass filter 0.5–40 Hz
        5. Notch filter at 60 Hz
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Pick only EEG channels (drop annotations, status channels)
    eeg_channels = [ch for ch in raw.ch_names
                    if not any(x in ch.upper() for x in ['ECG','EMG','EOG','--','VNS'])]
    raw.pick_channels(eeg_channels)

    # Set average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # Bandpass filter
    raw.filter(
        BANDPASS_LOW, BANDPASS_HIGH,
        method='fir', fir_window='hamming',
        verbose=False
    )

    # Notch filter (US powerline)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    return raw, raw.info['sfreq']


def create_epochs(raw, seizure_intervals: list) -> tuple:
    """
    Slice continuous EEG into 4-second epochs with 50% overlap.
    Labels each epoch as seizure (1) or non-seizure (0).

    Returns:
        epochs: np.ndarray shape (n_epochs, n_channels, n_samples)
        labels: np.ndarray shape (n_epochs,)
    """
    data = raw.get_data()             # (n_channels, n_total_samples)
    sfreq = raw.info['sfreq']
    n_channels, n_total = data.shape
    epochs, labels = [], []

    start = 0
    while start + EPOCH_SAMPLES <= n_total:
        end = start + EPOCH_SAMPLES
        epoch = data[:, start:end]

        # Time in seconds for this epoch
        t_start = start / sfreq
        t_end   = end   / sfreq

        # Label: seizure if any overlap with annotated seizure interval
        label = 0
        for sz_start, sz_end in seizure_intervals:
            if t_start < sz_end and t_end > sz_start:
                label = 1
                break

        epochs.append(epoch)
        labels.append(label)
        start += STEP_SAMPLES

    return np.array(epochs, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_patient_data(data_dir: str, patient: str = 'chb01') -> tuple:
    """
    Main entry point. Load all EDF files for a patient.

    Args:
        data_dir: path to folder containing chb01 files
                  e.g. C:/Users/gajos/OneDrive/NeuroScanAI
        patient:  patient ID prefix, default 'chb01'

    Returns:
        all_epochs: (N, C, T) array
        all_labels: (N,) array
        meta: dict with channel names, sfreq, class counts
    """
    data_dir = Path(data_dir)
    summary_path = data_dir / f'{patient}-summary.txt'
    seizure_map = parse_summary_file(str(summary_path))

    all_epochs, all_labels = [], []
    channel_names = None

    edf_files = sorted(data_dir.glob(f'{patient}_*.edf'))
    if not edf_files:
        raise FileNotFoundError(f'No EDF files found in {data_dir} for patient {patient}')

    print(f'Found {len(edf_files)} EDF files for {patient}')

    for edf_path in edf_files:
        filename = edf_path.name
        print(f'  Loading {filename}...', end=' ')

        # Get seizure intervals from summary or .seizures file
        seizure_intervals = seizure_map.get(filename, [])

        # Also check .edf.seizures file as backup
        seizure_file = str(edf_path) + '.seizures'
        if not seizure_intervals and os.path.exists(seizure_file):
            seizure_intervals = parse_seizure_file(seizure_file)

        try:
            raw, sfreq = load_and_preprocess_edf(str(edf_path))
            epochs, labels = create_epochs(raw, seizure_intervals)

            if channel_names is None:
                channel_names = raw.ch_names

            all_epochs.append(epochs)
            all_labels.append(labels)

            n_sz = labels.sum()
            print(f'{len(labels)} epochs ({n_sz} seizure, {len(labels)-n_sz} normal)')

        except Exception as e:
            print(f'FAILED: {e}')
            continue

    if not all_epochs:
        raise ValueError('No data loaded successfully')

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f'\nTotal: {X.shape[0]} epochs | '
          f'Seizure: {y.sum()} ({y.mean()*100:.1f}%) | '
          f'Normal: {(y==0).sum()}')

    meta = {
        'channel_names': channel_names,
        'sfreq': sfreq,
        'n_channels': X.shape[1],
        'n_samples': X.shape[2],
        'n_seizure': int(y.sum()),
        'n_normal': int((y==0).sum()),
        'patient': patient,
    }

    return X, y, meta
