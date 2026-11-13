"""Inference: loads the SAVED artifact (never hard-coded outputs)."""
import joblib
import pandas as pd
from .config import FEATURES, MODEL_PATH
from .features import prepare_X

HUMAN_REVIEW_NOTE = ("Advisory only. This score must be reviewed by a teacher/counsellor "
                     "and must never be used for grading, punishment or admission decisions.")
LABELS = {
    "attendance_rate": "overall attendance", "attendance_recent": "recent attendance",
    "assignment_completion": "assignment completion", "late_submission_rate": "late submissions",
    "lms_hours_per_week": "learning-platform activity", "prev_score_1": "oldest previous score",
    "prev_score_2": "middle previous score", "prev_score_3": "latest previous score",
    "avg_prev_score": "average previous score", "score_trend": "score trend",
    "attendance_change": "attendance change",
}


class RiskPredictor:
    def __init__(self, path=MODEL_PATH):
        self.art = joblib.load(path)
        self.pipe = self.art["pipeline"]

    def _proba(self, X: pd.DataFrame):
        return self.pipe.predict_proba(X)[:, 1]

    def predict(self, record: dict, top_k: int = 3) -> dict:
        df = pd.DataFrame([record])
        X = prepare_X(df)
        p = float(self._proba(X)[0])
        cut = self.art["band_cutoffs"]
        band = "High" if p >= cut["high"] else "Medium" if p >= cut["medium"] else "Low"

        # Local explanation by occlusion: replace one feature by the training
        # median and see how much the risk probability changes.
        med = self.art["train_medians"]
        effects = []
        for f in self.art["features"]:
            Xo = X.copy(); Xo[f] = med[f]
            effects.append((f, p - float(self._proba(Xo)[0])))
        effects.sort(key=lambda kv: -abs(kv[1]))
        signals = [{"signal": LABELS[f], "feature": f,
                    "value": None if pd.isna(X[f].iloc[0]) else round(float(X[f].iloc[0]), 3),
                    "effect": "raises risk" if d > 0 else "lowers risk",
                    "impact": round(abs(d), 3)} for f, d in effects[:top_k]]

        missing = [f for f in FEATURES if pd.isna(df[f].iloc[0])]
        notes = ["Probability is an estimate from a model trained on synthetic data; "
                 "treat it as a prompt for a conversation, not a verdict."]
        if missing:
            notes.append(f"Missing inputs filled with typical values: {missing}. "
                         "Confidence is lower.")
        return {"risk_probability": round(p, 3), "risk_band": band,
                "flagged_for_support": p >= cut["high"], "top_signals": signals,
                "model": self.art["model_name"], "uncertainty_notes": notes,
                "human_review_note": HUMAN_REVIEW_NOTE}
