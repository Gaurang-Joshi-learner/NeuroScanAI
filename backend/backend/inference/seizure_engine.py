"""
inference/seizure_engine.py
----------------------------
Loads trained seizure detection models and runs inference on uploaded EDF files.
Returns structured result with per-epoch predictions and overall risk score.
"""
import os, json
import numpy as np
import torch
import joblib
from pathlib import Path

# Import pipeline modules (path adjusted for backend context)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.seizure.preprocessing import load_and_preprocess_edf, create_epochs
from pipeline.seizure.features import extract_features
from pipeline.seizure.model import build_eegnet


class SeizureEngine:
    def __init__(self, models_dir: str = "models/seizure"):
        self.models_dir = models_dir
        self.eegnet   = None
        self.xgb      = None
        self.scaler   = None
        self.meta     = None
        self._loaded  = False

    def load(self):
        if self._loaded:
            return
        meta_path = os.path.join(self.models_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No trained model found at {self.models_dir}. "
                "Run: python -m pipeline.seizure.train --data_dir <path>"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        n_channels = self.meta["n_channels"]
        n_samples  = self.meta["n_samples"]

        # Load EEGNet
        self.eegnet = build_eegnet(n_channels, n_samples)
        self.eegnet.load_state_dict(
            torch.load(os.path.join(self.models_dir, "eegnet_best.pth"),
                       map_location="cpu")
        )
        self.eegnet.eval()

        # Load XGBoost
        self.xgb    = joblib.load(os.path.join(self.models_dir, "xgboost_model.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "xgboost_scaler.pkl"))
        self._loaded = True

    def predict(self, edf_path: str) -> dict:
        self.load()

        raw, sfreq = load_and_preprocess_edf(edf_path)
        epochs, _ = create_epochs(raw, seizure_intervals=[])  # no known labels for inference

        if len(epochs) == 0:
            return {"error": "No epochs could be extracted from this file"}

        # EEGNet inference
        with torch.no_grad():
            X_t = torch.FloatTensor(epochs)
            eegnet_probs = torch.softmax(self.eegnet(X_t), dim=1)[:, 1].numpy()

        # XGBoost inference
        features = extract_features(epochs, sfreq)
        xgb_probs = self.xgb.predict_proba(self.scaler.transform(features))[:, 1]

        # Ensemble
        ensemble_probs = 0.5 * eegnet_probs + 0.5 * xgb_probs

        # Build timeline (seconds)
        epoch_sec    = 4
        overlap      = 0.5
        step_sec     = epoch_sec * (1 - overlap)
        timeline = []
        for i, prob in enumerate(ensemble_probs.tolist()):
            t_start = i * step_sec
            timeline.append({
                "epoch_idx":   i,
                "time_start":  round(t_start, 2),
                "time_end":    round(t_start + epoch_sec, 2),
                "seizure_prob": round(prob, 4),
                "prediction":  "SEIZURE" if prob >= 0.5 else "NORMAL",
            })

        # Detected seizure events (consecutive high-prob epochs)
        seizure_epochs = [t for t in timeline if t["prediction"] == "SEIZURE"]
        overall_risk   = float(np.max(ensemble_probs))
        mean_risk      = float(np.mean(ensemble_probs))
        seizure_pct    = len(seizure_epochs) / max(len(timeline), 1) * 100

        return {
            "analysis_type":    "seizure_detection",
            "model_version":    "eegnet+xgboost-v1",
            "n_epochs":         len(timeline),
            "n_channels":       epochs.shape[1],
            "sfreq":            float(sfreq),
            "recording_duration_sec": round(len(timeline) * step_sec + epoch_sec, 1),
            "overall_risk_score":  round(overall_risk, 4),
            "mean_risk_score":     round(mean_risk, 4),
            "seizure_epoch_pct":   round(seizure_pct, 2),
            "risk_level":          self._risk_level(overall_risk),
            "n_seizure_epochs":    len(seizure_epochs),
            "timeline":            timeline,
            "channel_names":       self.meta.get("channel_names", []),
            "clinical_note": (
                "FOR RESEARCH USE ONLY. Not a clinical diagnostic tool. "
                "All findings must be reviewed by a qualified neurologist."
            )
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.75: return "HIGH"
        if score >= 0.50: return "MODERATE"
        if score >= 0.25: return "LOW"
        return "MINIMAL"


# Singleton instance
_seizure_engine = None

def get_seizure_engine(models_dir: str = "models/seizure") -> SeizureEngine:
    global _seizure_engine
    if _seizure_engine is None:
        _seizure_engine = SeizureEngine(models_dir)
    return _seizure_engine
