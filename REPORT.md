# Technical Report – Student Performance & At-Risk Prediction
Dasari Nishanth Reddy · Reg. No. 12302189

## 1. Framing
Advisory tool for teachers: at the end of week 8, flag students likely to score below 50
on the week-10 assessment so they can be offered support. A missed student (false
negative) is costlier than an unnecessary check-in (false positive), so recall is
prioritised and the decision threshold maximises F2 on the validation term.

## 2. Data and EDA
Synthetic generator (seed 42): 1,800 rows, 3 terms x 600 students, positive rate 23.4%.
Quality checks: missing values in `prev_score_1` (4.6%), `lms_hours_per_week` (4.4%),
`attendance_recent` (2.9%); 9 implausible LMS values (>30 h/week); no duplicate IDs.
Handling: median imputation fitted on training data only; LMS hours capped at 30; class
imbalance handled with balanced class weights and a tuned threshold.
See `reports/eda_distributions.png`.

## 3. Leakage analysis
Only week 1-8 information is used. `next_score` and `needs_support` are blocked by an
assertion in `train.py` and a unit test. Strongest feature correlates 0.57 with the target
(`avg_prev_score`), so no feature is a near-copy of the label. Terms are split chronologically
and no student appears in two terms. `gender` is excluded from the model.

## 4. Features
Raw: overall/recent attendance, assignment completion, late-submission rate, LMS hours,
three previous scores. Engineered: average previous score, score trend (latest minus oldest),
attendance change (recent minus overall). Rationale: engagement and recent performance are the
signals teachers can act on; protected attributes are excluded.

## 5. Models and results (held-out test term, 600 students, 127 positives)
| Model | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| Baseline: majority class | 0.00 | 0.00 | 0.00 | 0 | 0 | 127 | 473 |
| Baseline: rule on avg previous score | 0.48 | 0.87 | 0.62 | 110 | 118 | 17 | 355 |
| Logistic regression (selected) | 0.45 | 0.95 | 0.61 | 121 | 151 | 6 | 322 |
| Random forest | 0.48 | 0.95 | 0.64 | 121 | 133 | 6 | 340 |
| Hist. gradient boosting | 0.41 | 0.98 | 0.58 | 125 | 178 | 2 | 295 |

Selection used the validation term (highest F2: logistic regression 0.793 vs rule baseline 0.775).
Logistic regression also has the best ROC-AUC (0.928 vs 0.900 for the rule) and PR-AUC
(0.775 vs 0.705) on test, and is the easiest to explain. Honest reading: the learned models
raise recall (87% -> 95%) and ranking quality over the simple rule, but precision stays
below 0.5, so many flagged students will not actually fall below 50.

## 6. Error analysis
False positives (151): students whose averages are borderline (mean previous score about 56,
completion 0.75) and who then scored just above 50 – harmless if the response is a check-in.
False negatives (6): students with decent history (mean previous score about 63) whose next
score dropped by chance; no model with these inputs can foresee that noise.
Fairness audit (gender, test term): flag rate 45.3% (F) vs 45.4% (M); recall 0.92 vs 0.98,
a gap worth monitoring (small sample, 298 vs 302 students).

## 7. Inference and explainability
`app/api.py` loads `models/risk_model.joblib` and returns probability, band (High >= tuned
threshold, Medium >= half of it, else Low), the top three signals via occlusion (replace one
feature with the training median and measure the change), uncertainty notes and a human-review
note. Global importance (permutation): previous scores and assignment completion lead.

## 8. Limitations, bias, privacy, failure modes
- Synthetic data: performance will differ on real cohorts; retrain and revalidate before use.
- Representativeness: outcomes depend on the generator's assumptions; real data may encode
  historical bias (e.g. via attendance proxies).
- Precision below 0.5: use as a prompt for a human conversation, never an automated action.
- Occlusion explanations understate impact when probability is saturated or features are correlated.
- Privacy: no real or personal data is stored or committed; no secrets in the repo.
- Failure modes: missing inputs reduce confidence; drift in grading scales invalidates the threshold.
- Human oversight: every output states that a teacher must review it.

## 9. Reproducibility
Fixed seed (42), `requirements.txt`, saved artifact, 7 automated tests, commands in README.
