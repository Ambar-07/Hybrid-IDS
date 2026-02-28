"""
ML Anomaly Detector
-------------------
Uses Isolation Forest (unsupervised) to detect anomalous flows.
No labelled attack data needed for training — only normal traffic.

Output: anomaly_score (0.0 = normal, 1.0 = highly anomalous)
"""

import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import List
from sklearn.ensemble import IsolationForest
from engine.feature_extractor import FlowFeatures


@dataclass
class MLResult:
    anomaly_score: float    # 0.0 (normal) → 1.0 (anomaly)
    is_anomaly: bool
    raw_score: float        # raw Isolation Forest score (negative = anomaly)
    confidence: float       # same as anomaly_score for simplicity


class MLDetector:
    """
    Usage
    -----
    detector = MLDetector()
    detector.train(normal_flows)            # list of FlowFeatures
    result = detector.predict(flow)         # single FlowFeatures
    results = detector.predict_batch(flows) # list of FlowFeatures

    detector.save("models/isolation_forest.pkl")
    detector.load("models/isolation_forest.pkl")
    """

    def __init__(
        self,
        contamination: float = 0.05,   # expected % of outliers in training data
        n_estimators: int = 100,
        threshold: float = 0.55,       # anomaly_score threshold to flag
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators   = n_estimators
        self.threshold      = threshold
        self.random_state   = random_state
        self.model: IsolationForest = None
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, normal_flows: List[FlowFeatures]) -> "MLDetector":
        """Train ONLY on normal (benign) traffic."""
        X = np.array([f.normalized for f in normal_flows])
        print(f"[MLDetector] Training Isolation Forest on {len(X)} normal flows...")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.trained = True
        print("[MLDetector] Training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, ff: FlowFeatures) -> MLResult:
        """Score a single flow."""
        self._check_trained()
        X = ff.normalized.reshape(1, -1)
        raw_score = float(self.model.score_samples(X)[0])

        # Convert: Isolation Forest gives negative scores for anomalies
        # Typical range: -0.5 (anomaly) to 0.1 (normal)
        # We map to 0..1 where 1 = anomaly
        anomaly_score = self._scale_score(raw_score)
        is_anomaly = anomaly_score >= self.threshold

        return MLResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            raw_score=raw_score,
            confidence=anomaly_score,
        )

    def predict_batch(self, flows: List[FlowFeatures]) -> List[MLResult]:
        """Score multiple flows efficiently."""
        self._check_trained()
        X = np.array([f.normalized for f in flows])
        raw_scores = self.model.score_samples(X)

        return [
            MLResult(
                anomaly_score=self._scale_score(s),
                is_anomaly=self._scale_score(s) >= self.threshold,
                raw_score=float(s),
                confidence=self._scale_score(s),
            )
            for s in raw_scores
        ]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "models/isolation_forest.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.threshold}, f)
        print(f"[MLDetector] Model saved to {path}")

    def load(self, path: str = "models/isolation_forest.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model     = data["model"]
        self.threshold = data["threshold"]
        self.trained   = True
        print(f"[MLDetector] Model loaded from {path}")
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _scale_score(self, raw: float) -> float:
        """Map raw IF score to [0,1] where 1=anomaly."""
        # IF scores roughly range from -0.6 to +0.1
        normalized = (raw + 0.6) / 0.7          # shift to ~[0, 1]
        normalized = float(np.clip(normalized, 0, 1))
        return round(1.0 - normalized, 4)        # invert: high = anomaly

    def _check_trained(self):
        if not self.trained or self.model is None:
            raise RuntimeError("MLDetector is not trained. Call .train() or .load() first.")
