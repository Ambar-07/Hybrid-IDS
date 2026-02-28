"""
Load Model Utility
-------------------
Loads the trained Isolation Forest model for the Hybrid IDS.
This module can be imported by other scripts or run standalone to verify
that a trained model exists and is loadable.

Usage:
  from models.loadmodel import load_ids_model
  detector = load_ids_model()
"""

import os
import sys

# Ensure project root is on the path so engine imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.ml_detector import MLDetector

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "isolation_forest.pkl")


def load_ids_model(model_path: str = None) -> MLDetector:
    """
    Load a trained Isolation Forest MLDetector.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved .pkl model file.
        Defaults to  models/isolation_forest.pkl  inside the project.

    Returns
    -------
    MLDetector
        A trained MLDetector instance ready for prediction.
    """
    path = model_path or MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'.\n"
            "Please train first:\n"
            "  • CLI:       python main.py --train <your_training.csv>\n"
            "  • Dashboard: streamlit run ui/dashboard.py  →  ⚙️ Train Model"
        )

    detector = MLDetector()
    detector.load(path)
    print(f"[loadmodel] ✅ Isolation Forest model loaded from {path}")
    print(f"[loadmodel]    Anomaly threshold: {detector.threshold}")
    return detector


# ── Run standalone to verify ──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        detector = load_ids_model()
        print("[loadmodel] Model is ready for predictions.")
    except FileNotFoundError as e:
        print(f"[loadmodel] ⚠️  {e}")