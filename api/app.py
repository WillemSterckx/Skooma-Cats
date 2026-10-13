import json, os
from datetime import datetime
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

ART = os.environ.get("ARTIFACT_DIR", "artifacts")
booster = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
with open(os.path.join(ART, "schema.json"), encoding="utf-8") as f:
    schema = json.load(f)

NY = ZoneInfo("America/New_York")
STATIONS = schema["stations"]
RIDEABLE = {t: i for i, t in enumerate(schema["rideable_types"])}
FEATURES = schema["features"]

app = FastAPI(title="Citi Bike member/casual predictor")


class Ride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_station_id: str
    rideable_type: str
    started_at: datetime  # ISO 8601, e.g. 2026-10-12T08:30 (NYC local time if no offset)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/model")
def model_info():
    return {"model_version": schema["model_version"], "metrics": schema["metrics"]}


@app.get("/stations")
def stations():
    return [{"id": k, "name": v["name"]} for k, v in STATIONS.items()]


@app.get("/rideable_types")
def rideable_types():
    return list(RIDEABLE)


@app.post("/predict")
def predict(r: Ride):
    st = STATIONS.get(r.start_station_id)
    if st is None:
        raise HTTPException(422, "unknown start_station_id")
    if r.rideable_type not in RIDEABLE:
        raise HTTPException(422, f"rideable_type must be one of {list(RIDEABLE)}")
    t = r.started_at
    if t.tzinfo is not None:
        t = t.astimezone(NY)
    values = {
        "start_lat": st["lat"], "start_lng": st["lng"],
        "hour": t.hour, "isodow": t.isoweekday(), "month": t.month,
        "rideable_type": RIDEABLE[r.rideable_type],
    }
    x = np.array([[values[f] for f in FEATURES]], dtype=float)
    p_member = float(booster.predict(x)[0])
    label = schema["labels"]["1" if p_member >= 0.5 else "0"]
    return {"prediction": label, "probability_member": round(p_member, 4),
            "model_version": schema["model_version"]}
