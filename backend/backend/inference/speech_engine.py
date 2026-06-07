"""
inference/speech_engine.py
---------------------------
Loads trained speech decoder and runs inference on uploaded EEG files.
Returns predicted word/phoneme with confidence per trial.
"""
import os, json
import numpy as np
import torch
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.speech.preprocessing import PROMPTS, load_karaone_subject
from pipeline.speech.features import extract_features
from pipeline.speech.model import build_speech_eegnet


class SpeechEngine:
    def __init__(self, models_dir: str = "models/speech"):
        self.models_dir = models_dir
        self.eegnet  = None
        self.xgb     = None
        self.scaler  = None
        self.meta    = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        meta_path = os.path.join(self.models_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No trained speech model at {self.models_dir}. "
                "Run: python -m pipeline.speech.train --data_dir <karaone_path>"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)

        n_channels = self.meta["n_channels"]
        n_samples  = self.meta["n_samples"]
        n_classes  = len(PROMPTS)

        self.eegnet = build_speech_eegnet(n_channels, n_samples, n_classes)
        self.eegnet.load_state_dict(
            torch.load(os.path.join(self.models_dir, "speech_eegnet_best.pth"),
                       map_location="cpu")
        )
        self.eegnet.eval()
        self.xgb    = joblib.load(os.path.join(self.models_dir, "speech_xgboost.pkl"))
        self.scaler = joblib.load(os.path.join(self.models_dir, "speech_scaler.pkl"))
        self._loaded = True

    def predict(self, mat_path: str) -> dict:
        self.load()
        epochs, labels, meta = load_karaone_subject(mat_path)

        with torch.no_grad():
            logits = self.eegnet(torch.FloatTensor(epochs))
            eegnet_probs = torch.softmax(logits, dim=1).numpy()

        features, _ = extract_features(epochs, meta.get("sfreq", 256))
        xgb_probs = self.xgb.predict_proba(self.scaler.transform(features))

        ensemble_probs = 0.5 * eegnet_probs + 0.5 * xgb_probs

        trials = []
        for i in range(len(epochs)):
            pred_idx  = int(np.argmax(ensemble_probs[i]))
            true_idx  = int(labels[i]) if labels is not None else None
            top3 = sorted(
                [{"word": PROMPTS[j], "confidence": round(float(ensemble_probs[i][j]), 4)}
                 for j in range(len(PROMPTS))],
                key=lambda x: -x["confidence"]
            )[:3]
            trials.append({
                "trial_idx":       i,
                "predicted_word":  PROMPTS[pred_idx],
                "confidence":      round(float(ensemble_probs[i][pred_idx]), 4),
                "true_word":       PROMPTS[true_idx] if true_idx is not None else None,
                "correct":         (pred_idx == true_idx) if true_idx is not None else None,
                "top3":            top3,
            })

        correct = [t for t in trials if t["correct"] is True]
        accuracy = len(correct) / len(trials) if trials else 0

        return {
            "analysis_type": "speech_decoding",
            "model_version": "eegnet+xgboost-speech-v1",
            "n_trials":      len(trials),
            "n_classes":     len(PROMPTS),
            "vocabulary":    PROMPTS,
            "accuracy":      round(accuracy, 4),
            "chance_level":  round(1 / len(PROMPTS), 4),
            "trials":        trials,
            "clinical_note": "FOR RESEARCH USE ONLY. Not a clinical diagnostic tool."
        }


_speech_engine = None

def get_speech_engine(models_dir: str = "models/speech") -> SpeechEngine:
    global _speech_engine
    if _speech_engine is None:
        _speech_engine = SpeechEngine(models_dir)
    return _speech_engine
