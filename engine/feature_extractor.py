"""
Feature Extractor
-----------------
Takes raw network flow data (CSV row or dict) and returns
a normalized feature vector for the rule engine + ML model.

Compatible with CICIDS2017 column names out of the box.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


# ── The features we actually use ──────────────────────────────────────────────
SELECTED_FEATURES = [
    "duration",
    "protocol_type",      # will be encoded
    "src_port",
    "dst_port",
    "pkt_count",
    "byte_count",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "urg_flag_count",
    "avg_pkt_size",
    "active_mean",
    "idle_mean",
]

# CICIDS2017 → our internal name mapping
CICIDS_COLUMN_MAP = {
    "Flow Duration":             "duration",
    "Protocol":                  "protocol_type",
    "Source Port":               "src_port",
    "Destination Port":          "dst_port",
    "Total Fwd Packets":         "pkt_count",
    "Total Length of Fwd Packets": "byte_count",
    "Flow Bytes/s":              "flow_bytes_per_sec",
    "Flow Packets/s":            "flow_pkts_per_sec",
    "Fwd Packet Length Mean":    "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":    "bwd_pkt_len_mean",
    "SYN Flag Count":            "syn_flag_count",
    "ACK Flag Count":            "ack_flag_count",
    "FIN Flag Count":            "fin_flag_count",
    "RST Flag Count":            "rst_flag_count",
    "PSH Flag Count":            "psh_flag_count",
    "URG Flag Count":            "urg_flag_count",
    "Average Packet Size":       "avg_pkt_size",
    "Active Mean":               "active_mean",
    "Idle Mean":                 "idle_mean",
    "Label":                     "label",
}

PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, 6: 6, 17: 17, 1: 1}


@dataclass
class FlowFeatures:
    """Holds extracted features for one network flow."""
    raw: Dict[str, Any] = field(default_factory=dict)
    normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    label: str = "UNKNOWN"          # ground truth if available
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = ""


class FeatureExtractor:
    """
    Usage
    -----
    extractor = FeatureExtractor()
    extractor.fit(df_train)          # learns min/max for normalization
    features  = extractor.transform(row_dict)   # single flow
    df_out    = extractor.transform_df(df)      # whole dataframe
    """

    def __init__(self):
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}
        self.fitted = False

    # ── Fit (learn normalization stats from training data) ────────────────────
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        df = self._rename_columns(df)
        df = self._clean(df)
        for feat in SELECTED_FEATURES:
            if feat in df.columns and feat != "protocol_type":
                self._min[feat] = float(df[feat].min())
                self._max[feat] = float(df[feat].max())
        self.fitted = True
        print(f"[FeatureExtractor] Fitted on {len(df)} rows.")
        return self

    # ── Transform a single flow dict ──────────────────────────────────────────
    def transform(self, row: Dict[str, Any]) -> FlowFeatures:
        row = self._rename_row(row)
        row = self._fill_missing(row)

        ff = FlowFeatures(
            raw=row,
            label=str(row.get("label", "UNKNOWN")),
            src_ip=str(row.get("src_ip", "")),
            dst_ip=str(row.get("dst_ip", "")),
            src_port=int(row.get("src_port", 0)),
            dst_port=int(row.get("dst_port", 0)),
            protocol=str(row.get("protocol_type", "")),
        )

        vec = []
        for feat in SELECTED_FEATURES:
            val = float(row.get(feat, 0))
            if feat == "protocol_type":
                val = float(PROTOCOL_MAP.get(val, 0))
            else:
                val = self._normalize(feat, val)
            vec.append(val)

        ff.normalized = np.array(vec, dtype=np.float32)
        return ff

    # ── Transform a whole DataFrame ───────────────────────────────────────────
    def transform_df(self, df: pd.DataFrame):
        df = self._rename_columns(df)
        df = self._clean(df)
        return [self.transform(row) for row in df.to_dict(orient="records")]

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        return df.rename(columns=CICIDS_COLUMN_MAP)

    def _rename_row(self, row: Dict) -> Dict:
        return {CICIDS_COLUMN_MAP.get(k.strip(), k.strip()): v for k, v in row.items()}

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df

    def _fill_missing(self, row: Dict) -> Dict:
        for feat in SELECTED_FEATURES:
            if feat not in row:
                row[feat] = 0
        return row

    def _normalize(self, feat: str, val: float) -> float:
        mn = self._min.get(feat, 0)
        mx = self._max.get(feat, 1)
        if mx == mn:
            return 0.0
        return float(np.clip((val - mn) / (mx - mn), 0, 1))
