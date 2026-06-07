"""
pipeline/seizure/model.py
--------------------------
EEGNet — a compact CNN designed specifically for EEG classification.
Paper: Lawhern et al. 2018 "EEGNet: A Compact Convolutional Neural Network
       for EEG-based Brain-Computer Interfaces"

Architecture:
  Block 1: Temporal convolution (learns frequency filters)
  Block 2: Depthwise spatial convolution (learns spatial filters per channel)
  Block 3: Separable convolution (learns temporal summary)
  Classifier: Fully connected → sigmoid

Works on CPU — designed to be lightweight (< 2K parameters for binary).
Also includes a gradient-boosted XGBoost model as a fallback/ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ── EEGNet (PyTorch) ─────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 2,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout: float = 0.5):
        """
        Args:
            n_channels: number of EEG channels
            n_samples:  time samples per epoch (sfreq * epoch_sec)
            n_classes:  output classes (2 for binary seizure detection)
            F1:         number of temporal filters
            D:          depth multiplier for depthwise conv
            F2:         number of pointwise filters
            dropout:    dropout rate
        """
        super().__init__()

        # Block 1: Temporal convolution
        # Kernel = sfreq/2 → captures half-second patterns
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, n_samples // 2), padding=(0, n_samples // 4), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise spatial convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3: Separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Calculate flattened size
        self._flat_size = self._get_flat_size(n_channels, n_samples, F1, D, F2)

        # Classifier
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self, n_channels, n_samples, F1, D, F2):
        """Dynamically compute the flattened feature size."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        """
        x: (batch, channels, samples) → add channel dim → (batch, 1, channels, samples)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs.numpy()


# ── XGBoost Model ────────────────────────────────────────
def build_xgboost(scale_pos_weight: float = 10.0) -> XGBClassifier:
    """
    XGBoost classifier tuned for imbalanced EEG seizure data.
    scale_pos_weight handles class imbalance — seizures are rare (~5-10%).

    Args:
        scale_pos_weight: ratio of normal:seizure epochs
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',     # area under precision-recall (better for imbalanced)
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


# ── Model builder ─────────────────────────────────────────
def build_eegnet(n_channels: int, n_samples: int, n_classes: int = 2) -> EEGNet:
    return EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes)
