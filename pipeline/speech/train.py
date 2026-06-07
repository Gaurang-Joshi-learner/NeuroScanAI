"""
pipeline/speech/train.py
-------------------------
Training pipeline for imagined speech classification on KaraOne.

Usage:
    python -m pipeline.speech.train --data_dir "C:/path/to/karaone" --save_dir models/speech
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json

from pipeline.speech.preprocessing import load_karaone_dataset, PROMPTS
from pipeline.speech.features import extract_features
from pipeline.speech.model import build_speech_eegnet, build_speech_xgboost


def train_speech_eegnet(X_train, y_train, X_val, y_val,
                        n_channels, n_samples, n_classes=11,
                        epochs=60, batch_size=16, lr=5e-4, save_dir='models/speech'):
    os.makedirs(save_dir, exist_ok=True)

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_v  = torch.FloatTensor(X_val)
    y_v  = torch.LongTensor(y_val)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_v, y_v),   batch_size=batch_size)

    model = build_speech_eegnet(n_channels, n_samples, n_classes)
    print(f'EEGNet (speech) parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

        val_acc = correct / total
        history['train_loss'].append(np.mean(losses))
        history['val_acc'].append(val_acc)
        scheduler.step()

        print(f'Epoch {epoch+1:3d}/{epochs} | Loss: {np.mean(losses):.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'speech_eegnet_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f'Early stopping at epoch {epoch+1}')
                break

    model.load_state_dict(torch.load(os.path.join(save_dir, 'speech_eegnet_best.pth')))
    return model, history


def train_speech_xgboost(X_feat_train, y_train, X_feat_val, y_val,
                          n_classes=11, save_dir='models/speech'):
    os.makedirs(save_dir, exist_ok=True)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_feat_train)
    X_v  = scaler.transform(X_feat_val)

    xgb = build_speech_xgboost(n_classes)
    xgb.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=50)

    joblib.dump(xgb,    os.path.join(save_dir, 'speech_xgboost.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'speech_scaler.pkl'))
    return xgb, scaler


def run_training(data_dir, save_dir='models/speech', epochs=60):
    print('='*60)
    print('NeuroScanAI — Imagined Speech Decoding Training')
    print(f'Data: {data_dir}')
    print('='*60)

    X, y, meta = load_karaone_dataset(data_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

    print(f'Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}')
    print(f'Chance accuracy: {1/len(PROMPTS)*100:.1f}%')

    n_channels, n_samples = X.shape[1], X.shape[2]

    print('\n--- Training EEGNet ---')
    eegnet, history = train_speech_eegnet(
        X_train, y_train, X_val, y_val,
        n_channels, n_samples, epochs=epochs, save_dir=save_dir
    )

    print('\n--- Extracting features for XGBoost ---')
    X_feat_tr,   _ = extract_features(X_train, meta.get('sfreq', 256))
    X_feat_val2, _ = extract_features(X_val,   meta.get('sfreq', 256))
    X_feat_test, _ = extract_features(X_test,  meta.get('sfreq', 256))

    print('\n--- Training XGBoost ---')
    xgb, scaler = train_speech_xgboost(X_feat_tr, y_train, X_feat_val2, y_val,
                                        n_classes=len(PROMPTS), save_dir=save_dir)

    # Evaluate
    print('\n=== Final Test Evaluation ===')
    eegnet.eval()
    with torch.no_grad():
        logits = eegnet(torch.FloatTensor(X_test))
        eegnet_preds = logits.argmax(dim=1).numpy()

    xgb_preds = xgb.predict(scaler.transform(X_feat_test))

    print('\nEEGNet:')
    print(classification_report(y_test, eegnet_preds, target_names=PROMPTS))
    print('\nXGBoost:')
    print(classification_report(y_test, xgb_preds, target_names=PROMPTS))

    metrics = {
        'eegnet_accuracy':  float(accuracy_score(y_test, eegnet_preds)),
        'xgboost_accuracy': float(accuracy_score(y_test, xgb_preds)),
        'chance_accuracy':  round(1/len(PROMPTS), 4),
        'n_classes': len(PROMPTS),
        'prompts': PROMPTS,
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nEEGNet Accuracy:  {metrics["eegnet_accuracy"]*100:.1f}%')
    print(f'XGBoost Accuracy: {metrics["xgboost_accuracy"]*100:.1f}%')
    print(f'Chance:           {metrics["chance_accuracy"]*100:.1f}%')
    return eegnet, xgb, scaler, meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', default='models/speech')
    parser.add_argument('--epochs',   type=int, default=60)
    args = parser.parse_args()
    run_training(args.data_dir, args.save_dir, args.epochs)
