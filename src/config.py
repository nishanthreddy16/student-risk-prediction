"""Central configuration: paths, seed, feature lists.

Only information available BEFORE the prediction point is used as a feature.
Prediction point = end of week 8 of a term. Target = score on the NEXT
assessment (week 10 test) falling below the pass mark.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS_DIR = ROOT / "reports"

SEED = 42
PASS_MARK = 50.0          # next-assessment score below this => "needs support"
N_PER_TERM = 600
TERMS = [1, 2, 3]         # chronological: train=1, validation=2, test=3

FEATURES = [
    "attendance_rate",        # share of classes attended, weeks 1-8
    "attendance_recent",      # attendance in weeks 5-8
    "assignment_completion",  # share of assignments submitted, weeks 1-8
    "late_submission_rate",   # share of submitted work that was late
    "lms_hours_per_week",     # learning-platform activity
    "prev_score_1",           # oldest previous assessment
    "prev_score_2",
    "prev_score_3",           # most recent previous assessment
]
ENGINEERED = ["avg_prev_score", "score_trend", "attendance_change"]
ALL_FEATURES = FEATURES + ENGINEERED

# Present in the synthetic data ONLY for a fairness audit. Never a model input.
SENSITIVE = ["gender"]
TARGET = "needs_support"
LEAKY = ["next_score", TARGET]   # must never appear among model inputs
