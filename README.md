# Student Performance & At-Risk Prediction (advisory decision support)

Candidate: Dasari Nishanth Reddy · Reg. No. 12302189 · Role: AI & Machine Learning
Project: ALGIONS AI LABS internship assessment – Student Performance & At-Risk Prediction

An advisory model that flags students who may benefit from extra academic support
**before the next assessment**. It never triggers grading, punishment or admission decisions.

## Problem definition
| Item | Definition |
|---|---|
| Decision | Which students should a teacher check in with for extra support |
| Prediction unit | One student in one term |
| Prediction time | End of week 8 (only data from weeks 1-8 is used) |
| Horizon / target | `needs_support = 1` if the week-10 assessment score is below 50 |
| Features | attendance (overall, recent), assignment completion, late submissions, LMS hours, 3 previous scores, plus engineered average / trend / attendance change |
| Excluded | `next_score` and target (leakage); `gender` (sensitive – used only in the post-hoc fairness audit) |

## Data
Fully **synthetic** (no real student data) from `src/generate_data.py`, fixed seed 42,
1,800 rows = 3 terms x 600 students, ~23% positive class, with injected missing values
and outliers. Generator code is the dataset source; there is no external licence.

## Setup and reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_data     # creates data/students.csv
python -m src.train             # trains, evaluates, saves models/risk_model.joblib + reports/
python -m pytest -q             # preprocessing, leakage, inference and API tests
uvicorn app.api:app --reload    # open http://127.0.0.1:8000  (form)  or  /docs  (Swagger)
```
Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d \
'{"attendance_rate":0.6,"attendance_recent":0.5,"assignment_completion":0.4,
  "late_submission_rate":0.4,"lms_hours_per_week":2,"prev_score_1":45,
  "prev_score_2":40,"prev_score_3":38}'
```
The response contains probability, advisory band (Low/Medium/High), top influencing
signals, uncertainty notes and a human-review note. It is computed by the saved model.

## Split and evaluation
Chronological: term 1 = train, term 2 = validation (model choice + threshold), term 3 = test
(used once). Metrics: precision, recall, F1, confusion matrix (+ F2, ROC-AUC, PR-AUC).
Results: `reports/metrics.json`, figures in `reports/*.png`, discussion in `REPORT.md`.

## Structure
```
src/config.py  generate_data.py  features.py  train.py  predict.py
app/api.py     tests/test_pipeline.py     reports/   models/   data/
```

## Known limitations
Synthetic data only, so scores show the pipeline works, not real-world accuracy.
Single-feature explanations understate impact when probability is already saturated.
No live deployment: demo runs locally.

## AI assistance disclosure
This project was built with help from an AI assistant (Claude). Keep this section
accurate and mirror it in the submission form's AI-usage answer.
