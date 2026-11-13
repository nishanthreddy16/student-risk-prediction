import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config as C
from src.features import prepare_X
from src.generate_data import generate
from src.predict import RiskPredictor

SAMPLE = dict(attendance_rate=0.6, attendance_recent=0.5, assignment_completion=0.4,
              late_submission_rate=0.4, lms_hours_per_week=2, prev_score_1=45,
              prev_score_2=40, prev_score_3=38)


def test_generator_is_reproducible():
    a, b = generate(seed=1, n_per_term=50), generate(seed=1, n_per_term=50)
    pd.testing.assert_frame_equal(a, b)


def test_no_leakage_or_sensitive_features():
    assert not set(C.ALL_FEATURES) & set(C.LEAKY)
    assert not set(C.ALL_FEATURES) & set(C.SENSITIVE)


def test_split_is_chronological_and_disjoint():
    df = generate(n_per_term=40)
    ids = [set(df.loc[df.term == t, "student_id"]) for t in C.TERMS]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2])


def test_preprocessing_handles_missing_and_outliers():
    rec = pd.DataFrame([{**SAMPLE, "prev_score_1": None, "lms_hours_per_week": 500}])
    X = prepare_X(rec)
    assert list(X.columns) == C.ALL_FEATURES
    assert X["lms_hours_per_week"].iloc[0] == 30.0


def test_missing_column_is_rejected():
    with pytest.raises(ValueError):
        prepare_X(pd.DataFrame([{"attendance_rate": 0.9}]))


def test_prediction_comes_from_saved_model_and_orders_risk():
    p = RiskPredictor()
    weak, strong = p.predict(SAMPLE), p.predict({**SAMPLE, "attendance_rate": 0.98,
        "attendance_recent": 0.98, "assignment_completion": 0.97, "late_submission_rate": 0.02,
        "lms_hours_per_week": 9, "prev_score_1": 85, "prev_score_2": 88, "prev_score_3": 90})
    assert weak["risk_probability"] > strong["risk_probability"]
    assert weak["risk_band"] == "High" and strong["risk_band"] == "Low"
    assert "human_review_note" in weak and len(weak["top_signals"]) == 3


def test_api_predict_and_validation():
    from app.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=SAMPLE)
    assert r.status_code == 200 and 0 <= r.json()["risk_probability"] <= 1
    assert c.post("/predict", json={**SAMPLE, "attendance_rate": 7}).status_code == 422


def test_home_page_served():
    from app.api import app
    r = TestClient(app).get("/")
    assert r.status_code == 200 and "Student Support Advisory" in r.text
