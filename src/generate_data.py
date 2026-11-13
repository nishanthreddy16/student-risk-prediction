"""Reproducible synthetic data generator (no real student data is used).

Each row = one student in one term, observed at the end of week 8.
`next_score` (week-10 assessment) is generated AFTER, and only from, a latent
ability plus engagement signals; it is kept only to derive the target.
"""
import numpy as np
import pandas as pd
from .config import DATA_PATH, N_PER_TERM, PASS_MARK, SEED, TERMS


def generate(seed: int = SEED, n_per_term: int = N_PER_TERM) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for term in TERMS:
        n = n_per_term
        ability = rng.normal(0, 1, n)                       # latent, unobserved
        engagement = np.clip(0.6 * ability + rng.normal(0, 0.8, n), -3, 3)

        attendance = np.clip(0.82 + 0.07 * engagement + rng.normal(0, 0.07, n), 0.2, 1.0)
        attendance_recent = np.clip(attendance + rng.normal(0, 0.06, n)
                                    + 0.03 * np.minimum(engagement, 0), 0.1, 1.0)
        completion = np.clip(0.78 + 0.09 * engagement + rng.normal(0, 0.09, n), 0.05, 1.0)
        late = np.clip(0.18 - 0.05 * engagement + rng.normal(0, 0.07, n), 0.0, 1.0)
        lms = np.clip(5.0 + 1.6 * engagement + rng.normal(0, 1.5, n), 0.0, None)

        base = 62 + 11 * ability
        prev1 = np.clip(base + rng.normal(0, 7, n), 0, 100)
        prev2 = np.clip(base + 0.6 * engagement + rng.normal(0, 7, n), 0, 100)
        prev3 = np.clip(base + 1.2 * engagement + rng.normal(0, 7, n), 0, 100)

        next_score = np.clip(
            base + 3.5 * engagement
            + 12 * (completion - 0.78) + 10 * (attendance - 0.82)
            + rng.normal(0, 8, n), 0, 100)

        df = pd.DataFrame({
            "student_id": [f"T{term}-{i:04d}" for i in range(n)],
            "term": term,
            "gender": rng.choice(["F", "M"], n),   # audit only, independent of outcome
            "attendance_rate": attendance,
            "attendance_recent": attendance_recent,
            "assignment_completion": completion,
            "late_submission_rate": late,
            "lms_hours_per_week": lms,
            "prev_score_1": prev1, "prev_score_2": prev2, "prev_score_3": prev3,
            "next_score": next_score,
        })
        # Realistic data-quality issues: missing values and a few outliers
        for col, p in [("prev_score_1", 0.04), ("lms_hours_per_week", 0.05),
                       ("attendance_recent", 0.03)]:
            df.loc[rng.random(n) < p, col] = np.nan
        out = rng.random(n) < 0.01
        df.loc[out, "lms_hours_per_week"] = df.loc[out, "lms_hours_per_week"].fillna(5) * 6
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["needs_support"] = (data["next_score"] < PASS_MARK).astype(int)
    return data


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}; positive rate = {df['needs_support'].mean():.3f}")


if __name__ == "__main__":
    main()
