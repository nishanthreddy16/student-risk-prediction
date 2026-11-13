"""Feature engineering shared by training and inference (single source of truth)."""
import numpy as np
import pandas as pd
from .config import ALL_FEATURES, FEATURES


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev = out[["prev_score_1", "prev_score_2", "prev_score_3"]]
    out["avg_prev_score"] = prev.mean(axis=1)
    # change from oldest to most recent assessment (NaN-safe)
    out["score_trend"] = out["prev_score_3"] - out["prev_score_1"]
    out["attendance_change"] = out["attendance_recent"] - out["attendance_rate"]
    return out


def clip_outliers(df: pd.DataFrame, caps: dict | None = None) -> pd.DataFrame:
    """Cap lms_hours_per_week at a plausible ceiling (e.g. 30 h/week)."""
    out = df.copy()
    cap = (caps or {}).get("lms_hours_per_week", 30.0)
    out["lms_hours_per_week"] = out["lms_hours_per_week"].clip(upper=cap)
    return out


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = clip_outliers(add_engineered(df[FEATURES]))
    return X[ALL_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
