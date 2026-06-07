"""
pipeline/speech/model.py
-------------------------
EEGNet adapted for 11-class imagined speech classification.
Same architecture as seizure model but with softmax output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xgboost import XGBClassifier

PROMPTS = ['/iy/', '/uw/', '/piy/', '/tiy/', '/diy/',
           '/m/', '/n/', 'pat', 'pot', 'knew', 'gnaw']


class EEGNetSpeech(nn.Module):
    def __init__(self, n_channels: int, n_samples: int,
                 n_classes: int = 11, F1: int = 8, D: int = 2,
                 F2: int = 16, dropout: float = 0.5):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, n_samples // 2),
                      padding=(0, n_samples // 4), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1*D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(F1*D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        self._flat = self._get_flat(n_channels, n_samples, F1, D, F2)
        self.classifier = nn.Linear(self._flat, n_classes)

    def _get_flat(self, nc, ns, F1, D, F2):
        with torch.no_grad():
            x = torch.zeros(1, 1, nc, ns)
            x = self.block1(x); x = self.block2(x); x = self.block3(x)
            return x.numel()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_speech_xgboost(n_classes: int = 11) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


def build_speech_eegnet(n_channels, n_samples, n_classes=11):
    return EEGNetSpeech(n_channels, n_samples, n_classes)
