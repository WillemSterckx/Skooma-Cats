"""Train the member-vs-casual model.

Usage:  python training/train.py --out api/artifacts
Env:    DATA_GLOB   glob of cleaned parquet files (required)
        SAMPLE_ROWS max training rows to sample (default 8,000,000)
        MIN_AUC     fail the run if test AUC is below this (default 0.60)
        TRAIN_FROM  only use rides on/after this date (default 2021-02-01: current
                    schema, bike type recorded, same station naming)
"""
import argparse, glob, json, os, sys
from datetime import datetime, timezone

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# Only features that are known when a trip STARTS (no end station / duration).
FEATURES = ["start_lat", "start_lng", "hour", "isodow", "month", "rideable_type"]
CATEGORICAL = ["rideable_type"]
LABELS = {"0": "casual", "1": "member"}

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
args = p.parse_args()

data_glob = os.environ.get("DATA_GLOB")
if not data_glob or not glob.glob(data_glob, recursive=True):
    sys.exit(f"DATA_GLOB not set or matches no files: {data_glob!r}")
sample_rows = int(os.environ.get("SAMPLE_ROWS", 8_000_000))
min_auc = float(os.environ.get("MIN_AUC", 0.60))
train_from = os.environ.get("TRAIN_FROM", "2021-02-01")

con = duckdb.connect()
# Drop rides with unknown rider/bike type (kept by 03_prepare as "unknown") and old rides.
src = f"""(SELECT * FROM read_parquet('{data_glob.replace(chr(92), '/')}')
         WHERE member_casual::VARCHAR IN ('member', 'casual')
           AND rideable_type::VARCHAR <> 'unknown'
           AND started_at >= TIMESTAMP '{train_from}')"""

total, tmax = con.sql(f"SELECT count(*), max(started_at) FROM {src}").fetchone()
print(f"rows={total:,} latest ride={tmax}")

# --- station table for the API (stations active in the last 6 months) ---
stations = con.sql(f"""
    SELECT start_station_id AS id, any_value(start_station_name) AS name,
           median(start_lat) AS lat, median(start_lng) AS lng, count(*) AS n
    FROM {src}
    WHERE started_at >= TIMESTAMP '{tmax}' - INTERVAL 6 MONTH
      AND start_station_id <> 'no_station' AND start_station_name <> 'no_station'
    GROUP BY 1 HAVING count(*) >= 50 ORDER BY name
""").df()
print(f"active stations: {len(stations)}")

# --- rideable types ---
rtypes = con.sql(f"""
    SELECT rideable_type::VARCHAR AS rideable_type FROM {src}
    GROUP BY 1 ORDER BY count(*) DESC
""").df()["rideable_type"].tolist()
code = {t: i for i, t in enumerate(rtypes)}

frac = min(1.0, sample_rows / total)

def load(where):
    df = con.sql(f"""
        SELECT start_lat, start_lng,
               hour(started_at)::INT AS hour,
               date_part('isodow', started_at)::INT AS isodow,
               month(started_at)::INT AS month,
               rideable_type::VARCHAR AS rideable_type,
               (member_casual::VARCHAR = 'member')::INT AS y
        FROM {src}
        WHERE {where} AND random() < {frac}
    """).df()
    df["rideable_type"] = df["rideable_type"].map(code).astype("int32")
    return df[FEATURES], df["y"].to_numpy()

# time-based split: train | valid (2 months) | test (latest month)
t_test  = f"TIMESTAMP '{tmax}' - INTERVAL 1 MONTH"
t_valid = f"TIMESTAMP '{tmax}' - INTERVAL 3 MONTH"
Xtr, ytr = load(f"started_at < {t_valid}")
Xva, yva = load(f"started_at >= {t_valid} AND started_at < {t_test}")
Xte, yte = load(f"started_at >= {t_test}")
print(f"train={len(Xtr):,} valid={len(Xva):,} test={len(Xte):,} member share={ytr.mean():.3f}")

dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CATEGORICAL)
dva = lgb.Dataset(Xva, yva, reference=dtr)
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1)
booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

proba = booster.predict(Xte, num_iteration=booster.best_iteration)
metrics = dict(
    test_auc=float(roc_auc_score(yte, proba)),
    test_pr_auc=float(average_precision_score(yte, proba)),
    test_accuracy=float(accuracy_score(yte, proba > 0.5)),
    baseline_accuracy=float(max(yte.mean(), 1 - yte.mean())),
    best_iteration=int(booster.best_iteration),
    train_rows=int(len(Xtr)),
)
print(json.dumps(metrics, indent=2))

os.makedirs(args.out, exist_ok=True)
booster.save_model(os.path.join(args.out, "model.txt"), num_iteration=booster.best_iteration)
schema = dict(
    model_version=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    features=FEATURES, labels=LABELS, rideable_types=rtypes, metrics=metrics,
    stations={r.id: dict(name=r.name, lat=float(r.lat), lng=float(r.lng))
              for r in stations.itertuples()},
)
with open(os.path.join(args.out, "schema.json"), "w", encoding="utf-8") as f:
    json.dump(schema, f)

if metrics["test_auc"] < min_auc:
    sys.exit(f"Test AUC {metrics['test_auc']:.3f} below MIN_AUC {min_auc}; failing the run.")
