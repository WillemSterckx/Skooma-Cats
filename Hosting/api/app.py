import json, math, os
from datetime import datetime
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.environ.get("ARTIFACT_DIR", os.path.join(HERE, "artifacts"))
booster = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
with open(os.path.join(ART, "schema.json"), encoding="utf-8") as f:
    schema = json.load(f)

NY = ZoneInfo("America/New_York")
STATIONS = schema["stations"]
MEMBER_TYPES = schema["member_types"]
FEATURES = schema["features"]
MIN_DURATION, MAX_DURATION = 1.0, 24 * 60.0   # same rules as the cleaning step


def to_ny(t: datetime) -> datetime:
    """Naive times are treated as New York local time; aware times are converted."""
    return t.astimezone(NY).replace(tzinfo=None) if t.tzinfo is not None else t


app = FastAPI(title="Citi Bike electric-bike predictor")


class Ride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_station_id: str
    member_casual: str      # "member" or "casual"
    started_at: datetime    # ISO 8601, e.g. 2026-10-12T08:30 (New York time if no offset)
    ended_at: datetime      # same format; must be 1 minute to 24 hours after started_at


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/model")
def model_info():
    return {"model_version": schema["model_version"], "trained_until": schema["trained_until"],
            "metrics": schema["metrics"]}


@app.get("/stations")
def stations():
    return [{"id": k, "name": v["name"]} for k, v in STATIONS.items()]


@app.post("/predict")
def predict(r: Ride):
    st = STATIONS.get(r.start_station_id)
    if st is None:
        raise HTTPException(422, "unknown start_station_id")
    if r.member_casual not in MEMBER_TYPES:
        raise HTTPException(422, f"member_casual must be one of {MEMBER_TYPES}")

    t = to_ny(r.started_at)
    duration = (to_ny(r.ended_at) - t).total_seconds() / 60
    if not (MIN_DURATION <= duration <= MAX_DURATION):
        raise HTTPException(422, "ended_at must be between 1 minute and 24 hours after started_at")

    dow = t.isoweekday()   # 1 = Monday .. 7 = Sunday, matches DuckDB isodow
    values = {
        "start_lat": st["lat"],
        "start_lng": st["lng"],
        "hour_sin": math.sin(2 * math.pi * t.hour / 24),
        "hour_cos": math.cos(2 * math.pi * t.hour / 24),
        "dow_sin":  math.sin(2 * math.pi * dow / 7),
        "dow_cos":  math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * (t.month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (t.month - 1) / 12),
        "is_weekend": 1 if dow >= 6 else 0,
        "year": t.year,
        "is_member": 1 if r.member_casual == "member" else 0,
        "duration_min": duration,
        "log_duration_min": math.log1p(max(duration, 0.0)),
        "station_elec_rate": st["elec_rate"],
        "station_rides_log": st["rides_log"],
    }
    x = np.array([[values[f] for f in FEATURES]], dtype=float)
    p_electric = float(booster.predict(x)[0])
    label = schema["labels"]["1" if p_electric >= 0.5 else "0"]
    return {"prediction": label, "probability_electric": round(p_electric, 4),
            "model_version": schema["model_version"]}