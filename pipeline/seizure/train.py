"""
pipeline/seizure/train.py
--------------------------
Complete training pipeline for seizure detection.
Trains both EEGNet and XGBoost, saves best models, produces metrics.

Usage:
    python -m pipeline.seizure.train --data_dir "C:/Users/gajos/OneDrive/NeuroScanAI" --patient chb01
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score)
import joblib
import json
from pathlib import Path

from pipeline.seizure.preprocessing import load_patient_data
from pipeline.seizure.features import extract_features
from pipeline.seizure.model import build_eegnet, build_xgboost


def get_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Handle class imbalance by oversampling minority (seizure) class."""
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )


def train_eegnet(X_train, y_train, X_val, y_val, n_channels, n_samples,
                 epochs=50, batch_size=32, lr=1e-3, save_dir='models'):
    """Train EEGNet with early stopping."""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_v  = torch.FloatTensor(X_val)
    y_v  = torch.LongTensor(y_val)

    # Weighted sampler for imbalanced data
    sampler = get_weighted_sampler(y_train)
    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_v, y_v)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = build_eegnet(n_channels, n_samples)
    print(f'EEGNet parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Class weights for loss (extra push for seizure class)
    n_normal  = (y_train == 0).sum()
    n_seizure = (y_train == 1).sum()
    class_weight = torch.FloatTensor([1.0, n_normal / max(n_seizure, 1)])
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb)
                loss = criterion(out, yb)
                val_losses.append(loss.item())
                probs = torch.softmax(out, dim=1)[:, 1]
                all_probs.extend(probs.numpy())
                all_labels.extend(yb.numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_auc    = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1:3d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val AUC: {val_auc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'eegnet_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best weights
    model.load_state_dict(torch.load(os.path.join(save_dir, 'eegnet_best.pth')))
    return model, history


def train_xgboost(X_feat_train, y_train, X_feat_val, y_val, save_dir='models'):
    """Train XGBoost on handcrafted features."""
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_feat_train)
    X_v_scaled  = scaler.transform(X_feat_val)

    n_normal  = (y_train == 0).sum()
    n_seizure = (y_train == 1).sum()
    spw = n_normal / max(n_seizure, 1)

    xgb = build_xgboost(scale_pos_weight=spw)
    xgb.fit(
        X_tr_scaled, y_train,
        eval_set=[(X_v_scaled, y_val)],
        verbose=50
    )

    joblib.dump(xgb,    os.path.join(save_dir, 'xgboost_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'xgboost_scaler.pkl'))
    print(f'XGBoost saved to {save_dir}')
    return xgb, scaler


def evaluate_model(model, X_test, y_test, model_type='eegnet',
                   scaler=None, threshold=0.5):
    """Compute and print full evaluation metrics."""
    if model_type == 'eegnet':
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_test)
            probs = torch.softmax(model(X_t), dim=1)[:, 1].numpy()
    else:
        X_scaled = scaler.transform(X_test)
        probs = model.predict_proba(X_scaled)[:, 1]

    preds = (probs >= threshold).astype(int)

    print(f'\n=== {model_type.upper()} Evaluation ===')
    print(classification_report(y_test, preds, target_names=['Normal', 'Seizure']))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))

    if len(set(y_test)) > 1:
        auc  = roc_auc_score(y_test, probs)
        aupr = average_precision_score(y_test, probs)
        print(f'ROC-AUC: {auc:.4f}')
        print(f'PR-AUC (AUCPR): {aupr:.4f}  ← key metric for imbalanced data')

    return probs


def run_training(data_dir: str, patient: str = 'chb01', save_dir: str = 'models',
                 test_size: float = 0.2, val_size: float = 0.1,
                 epochs: int = 50):
    """Full training pipeline."""
    print('='*60)
    print(f'NeuroScanAI — Seizure Detection Training')
    print(f'Patient: {patient} | Data: {data_dir}')
    print('='*60)

    # 1. Load and preprocess
    X, y, meta = load_patient_data(data_dir, patient)
    print(f'\nData shape: {X.shape}')

    # 2. Train/val/test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        random_state=42, stratify=y_temp
    )
    print(f'\nTrain: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}')

    os.makedirs(save_dir, exist_ok=True)

    # Save metadata
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # 3. Train EEGNet (on raw epochs)
    print('\n--- Training EEGNet ---')
    n_channels = X.shape[1]
    n_samples  = X.shape[2]
    eegnet, history = train_eegnet(
        X_train, y_train, X_val, y_val,
        n_channels, n_samples, epochs=epochs, save_dir=save_dir
    )

    # 4. Extract features and train XGBoost
    print('\n--- Extracting features for XGBoost ---')
    X_feat_train = extract_features(X_train, meta['sfreq'])
    X_feat_val   = extract_features(X_val,   meta['sfreq'])
    X_feat_test  = extract_features(X_test,  meta['sfreq'])

    print('\n--- Training XGBoost ---')
    xgb, scaler = train_xgboost(X_feat_train, y_train, X_feat_val, y_val, save_dir)

    # 5. Evaluate both on test set
    print('\n--- Final Evaluation on Test Set ---')
    eegnet_probs = evaluate_model(eegnet, X_test, y_test, 'eegnet')
    xgb_probs    = evaluate_model(xgb, X_feat_test, y_test, 'xgboost', scaler)

    # 6. Ensemble (average probabilities)
    ensemble_probs = 0.5 * eegnet_probs + 0.5 * xgb_probs
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    print('\n=== ENSEMBLE Evaluation ===')
    print(classification_report(y_test, ensemble_preds, target_names=['Normal', 'Seizure']))

    print(f'\nModels saved to: {save_dir}/')
    return eegnet, xgb, scaler, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to chb01 folder')
    parser.add_argument('--patient',  default='chb01')
    parser.add_argument('--save_dir', default='models/seizure')
    parser.add_argument('--epochs',   type=int, default=50)
    args = parser.parse_args()

    run_training(args.data_dir, args.patient, args.save_dir, epochs=args.epochs)
