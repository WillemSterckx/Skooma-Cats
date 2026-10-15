"""Train the "will this ride use an electric bike?" model.

Usage:  python Hosting/training/train.py --out Hosting/api/artifacts
Env:    DATA_GLOB   glob of cleaned parquet files (required)
        MIN_AUC     fail the run if test AUC is below this (default 0.60)
        TRAIN_FROM  only use rides on/after this date (default 2021-02-01)

Inputs at prediction time: start station, start time, end time, rider type (member/casual).
The end time becomes duration_min, so this model classifies a ride AFTER it has happened.
Only rides that start at a dock station are used ("no_station" rides are excluded).
"""
import argparse, glob, json, os, sys
from datetime import datetime, timezone

import duckdb
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# NOTE: this list is duplicated in api/schema.json; the API reads it from there.
FEATURES = [
    "start_lat", "start_lng",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "year",
    "is_member",
    "duration_min", "log_duration_min",
    "station_elec_rate", "station_rides_log",
]
LABELS = {"0": "classic_bike", "1": "electric_bike"}
MEMBER_TYPES = ["casual", "member"]

# smoothing strength for the station target-encoding: a station needs this many
# rides before its own e-bike rate starts to dominate the global prior.
SMOOTH_K = 50.0

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
args = p.parse_args()

data_glob = os.environ.get("DATA_GLOB")
if not data_glob or not glob.glob(data_glob, recursive=True):
    sys.exit(f"DATA_GLOB not set or matches no files: {data_glob!r}")
min_auc = float(os.environ.get("MIN_AUC", 0.60))
train_from = os.environ.get("TRAIN_FROM", "2021-02-01")

con = duckdb.connect()
con.sql(f"PRAGMA threads={max(1, (os.cpu_count() or 2) - 1)}")

src = f"""(SELECT * FROM read_parquet('{data_glob.replace(chr(92), '/')}')
         WHERE member_casual::VARCHAR IN ('member', 'casual')
           AND rideable_type::VARCHAR IN ('classic_bike', 'electric_bike')
           AND start_station_id::VARCHAR <> 'no_station'
           AND start_station_name::VARCHAR <> 'no_station'
           AND started_at >= TIMESTAMP '{train_from}')"""

total, tmax = con.sql(f"SELECT count(*), max(started_at) FROM {src}").fetchone()
print(f"rows={total:,} latest ride={tmax}")

t_test  = f"TIMESTAMP '{tmax}' - INTERVAL 1 MONTH"
t_valid = f"TIMESTAMP '{tmax}' - INTERVAL 3 MONTH"


def prior_for(where: str) -> float:
    return float(con.sql(
        f"SELECT avg((rideable_type::VARCHAR = 'electric_bike')::INT) "
        f"FROM {src} WHERE {where}"
    ).fetchone()[0])


def build_stats(table: str, where: str) -> None:
    """Materialise per-start-station e-bike counts for a time window."""
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT start_station_id::VARCHAR AS id,
               count(*)::DOUBLE AS n,
               sum((rideable_type::VARCHAR = 'electric_bike')::INT)::DOUBLE AS pos
        FROM {src}
        WHERE {where}
        GROUP BY 1
    """)


def load(where: str, table: str, prior: float):
    """Pull every ride in the window plus its (smoothed) station encoding."""
    prior_sql = f"CAST({prior!r} AS DOUBLE)"
    k_sql = f"CAST({float(SMOOTH_K)!r} AS DOUBLE)"
    return con.sql(f"""
        SELECT r.start_lat::DOUBLE                AS start_lat,
               r.start_lng::DOUBLE                AS start_lng,
               hour(r.started_at)::INT            AS hour,
               date_part('isodow', r.started_at)::INT AS isodow,
               month(r.started_at)::INT           AS month,
               year(r.started_at)::INT            AS year,
               (r.member_casual::VARCHAR = 'member')::INT AS is_member,
               r.duration_min::DOUBLE             AS duration_min,
               (r.rideable_type::VARCHAR = 'electric_bike')::INT AS y,
               COALESCE(
                   (s.pos + {prior_sql} * {k_sql}) / (s.n + {k_sql}),
                   {prior_sql}
               )::DOUBLE                          AS station_elec_rate,
               ln(COALESCE(s.n, 0) + 1)::DOUBLE   AS station_rides_log
        FROM {src} r
        LEFT JOIN {table} s ON s.id = r.start_station_id::VARCHAR
        WHERE {where}
    """).df()

def featurize(df):
    h = df["hour"].to_numpy(dtype=float)
    d = df["isodow"].to_numpy(dtype=float)
    m = df["month"].to_numpy(dtype=float)
    df["hour_sin"]  = np.sin(2 * np.pi * h / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * h / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * d / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * d / 7)
    df["month_sin"] = np.sin(2 * np.pi * (m - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (m - 1) / 12)
    df["is_weekend"] = (d >= 6).astype(int)
    df["log_duration_min"] = np.log1p(df["duration_min"].clip(lower=0.0))
    return df[FEATURES], df["y"].to_numpy()


PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=500,
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
    num_threads=max(1, (os.cpu_count() or 2) - 1),
)

# ---------------------------------------------------------------------------
# 1) Evaluation on a time-based split: train | valid (2 months) | test (1 month)
#    Station stats are built ONLY from the train window so valid/test are clean.
# ---------------------------------------------------------------------------
print("building station stats (train window) ...")
build_stats("st_train", f"started_at < {t_valid}")
prior_tr = prior_for(f"started_at < {t_valid}")

Xtr, ytr = featurize(load(f"started_at < {t_valid}", "st_train", prior_tr))
Xva, yva = featurize(load(f"started_at >= {t_valid} AND started_at < {t_test}", "st_train", prior_tr))
Xte, yte = featurize(load(f"started_at >= {t_test}", "st_train", prior_tr))
print(f"train={len(Xtr):,} valid={len(Xva):,} test={len(Xte):,} "
      f"electric share={ytr.mean():.3f}")

dtr = lgb.Dataset(Xtr, ytr)
dva = lgb.Dataset(Xva, yva, reference=dtr)
booster = lgb.train(
    PARAMS, dtr,
    num_boost_round=3000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
)

proba = booster.predict(Xte, num_iteration=booster.best_iteration)
metrics = dict(
    test_auc=float(roc_auc_score(yte, proba)),
    test_pr_auc=float(average_precision_score(yte, proba)),
    test_accuracy=float(accuracy_score(yte, proba > 0.5)),
    baseline_accuracy=float(max(yte.mean(), 1 - yte.mean())),
    best_iteration=int(booster.best_iteration),
    train_rows=int(len(Xtr)),
    valid_rows=int(len(Xva)),
    test_rows=int(len(Xte)),
)
print(json.dumps(metrics, indent=2))

if metrics["test_auc"] < min_auc:
    sys.exit(f"Test AUC {metrics['test_auc']:.3f} below MIN_AUC {min_auc}; failing the run.")

# ---------------------------------------------------------------------------
# 2) Retrain on ALL data for the hosted model.
#    Station encoding is rebuilt on the full history so that serving-time
#    features exactly match training-time features.
# ---------------------------------------------------------------------------
print("building station stats (full history) ...")
build_stats("st_all", "TRUE")
prior_all = prior_for("TRUE")

Xall, yall = featurize(load("TRUE", "st_all", prior_all))
print(f"retraining on all {len(Xall):,} rows ...")

# Scale the number of rounds by the extra data added (valid + test folds).
n_final = max(50, int(round(booster.best_iteration * len(Xall) / max(1, len(Xtr)))))
print(f"final boosting rounds: {n_final}")

final = lgb.train(PARAMS, lgb.Dataset(Xall, yall), num_boost_round=n_final)

os.makedirs(args.out, exist_ok=True)
final.save_model(os.path.join(args.out, "model.txt"))

# Stations the API will accept, each carrying the exact encoding values the
# final model was trained with.
prior_all_sql = f"CAST({prior_all!r} AS DOUBLE)"
k_sql = f"CAST({float(SMOOTH_K)!r} AS DOUBLE)"
stations = con.sql(f"""
    SELECT s.id::VARCHAR                       AS id,
           any_value(r.start_station_name)     AS name,
           median(r.start_lat)::DOUBLE         AS lat,
           median(r.start_lng)::DOUBLE         AS lng,
           ((s.pos + {prior_all_sql} * {k_sql}) / (s.n + {k_sql}))::DOUBLE AS elec_rate,
           ln(s.n + 1)::DOUBLE                 AS rides_log
    FROM {src} r
    JOIN st_all s ON s.id = r.start_station_id::VARCHAR
    WHERE r.started_at >= TIMESTAMP '{tmax}' - INTERVAL 6 MONTH
    GROUP BY s.id, s.pos, s.n
    HAVING count(*) >= 50
    ORDER BY name
""").df()
print(f"active stations: {len(stations)}")

schema = dict(
    model_version=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    features=FEATURES,
    labels=LABELS,
    member_types=MEMBER_TYPES,
    metrics=metrics,
    final_rounds=int(n_final),
    final_rows=int(len(Xall)),
    trained_from=train_from,
    trained_until=str(tmax),
    stations={
        r.id: dict(name=r.name, lat=float(r.lat), lng=float(r.lng),
                   elec_rate=float(r.elec_rate), rides_log=float(r.rides_log))
        for r in stations.itertuples()
    },
)
with open(os.path.join(args.out, "schema.json"), "w", encoding="utf-8") as f:
    json.dump(schema, f)

print("done.")