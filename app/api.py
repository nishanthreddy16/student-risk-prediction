"""FastAPI inference service.  Run: uvicorn app.api:app --reload"""
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.predict import RiskPredictor

app = FastAPI(title="Student At-Risk Advisory API", version="1.0")
_predictor: Optional[RiskPredictor] = None


def predictor() -> RiskPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = RiskPredictor()
        except Exception as exc:  # missing or incompatible artifact
            raise HTTPException(503, "Model could not be loaded. Run `python -m src.train` "
                                     f"on this machine first. Details: {exc}")
    return _predictor


class StudentRecord(BaseModel):
    attendance_rate: float = Field(ge=0, le=1)
    attendance_recent: Optional[float] = Field(default=None, ge=0, le=1)
    assignment_completion: float = Field(ge=0, le=1)
    late_submission_rate: float = Field(ge=0, le=1)
    lms_hours_per_week: Optional[float] = Field(default=None, ge=0)
    prev_score_1: Optional[float] = Field(default=None, ge=0, le=100)
    prev_score_2: float = Field(ge=0, le=100)
    prev_score_3: float = Field(ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor().art["model_name"]}


@app.post("/predict")
def predict(rec: StudentRecord):
    return predictor().predict(rec.model_dump())


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "index.html")
