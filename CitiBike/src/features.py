"""Shared feature code for the Citi Bike hourly-demand model.

Used by the model notebooks, the API and the retraining pipeline, so that
training and inference always build features in exactly the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CLEAN_DIR = Path("citibike_data/clean")
DEMAND_PATH = Path("citibike_data/hourly_demand.parquet")

TOP_N_STATIONS = 100          # model the busiest stations only (see 04 for why)
TARGET = "trips"
FEATURES = ["station", "hour", "dayofweek", "month", "year",
            "is_weekend", "is_holiday", "hour_sin", "hour_cos", "lat", "lng"]


def build_hourly_demand(clean_dir: Path = CLEAN_DIR, top_n: int = TOP_N_STATIONS) -> pd.DataFrame:
    """Trips per (station, hour) for the top-n stations, from ALL clean files.

    * Stations are identified by NAME: Citi Bike replaced every station ID in
      2021 (short integers became 'NNNN.NN' codes), while most names stayed
      the same (checked in 02).
    * The top-n is chosen on the LAST 12 months, so we model stations that exist now.
    * Hours with zero trips are added, but only from a station's first trip
      onwards (before that the station did not exist; those zeros would be fake).
    DuckDB does the counting out-of-core: 300M+ trips never sit in memory.
    """
    import duckdb
    src = f"read_parquet('{clean_dir.as_posix()}/**/*.parquet')"
    return duckdb.connect().execute(f"""
        WITH trips AS (
            SELECT start_station_name AS station, started_at, start_lat, start_lng
            FROM {src} WHERE start_station_name <> 'no_station'),
        last_ts AS (SELECT max(started_at) AS mx FROM trips),
        top AS (
            SELECT station, count(*) AS n, median(start_lat) AS lat, median(start_lng) AS lng
            FROM trips, last_ts WHERE started_at >= mx - INTERVAL 365 DAY
            GROUP BY station ORDER BY n DESC LIMIT {int(top_n)}),
        counts AS (
            SELECT station, date_trunc('hour', started_at) AS ts, count(*) AS trips
            FROM trips WHERE station IN (SELECT station FROM top) GROUP BY ALL),
        grid AS (
            SELECT station, unnest(generate_series(min(ts),
                   (SELECT date_trunc('hour', mx) FROM last_ts), INTERVAL 1 HOUR)) AS ts
            FROM counts GROUP BY station)
        SELECT g.station, g.ts, coalesce(c.trips, 0)::INTEGER AS trips, t.lat, t.lng
        FROM grid g JOIN top t USING (station) LEFT JOIN counts c USING (station, ts)
        ORDER BY g.ts, g.station""").df()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features from 'ts'. Only information known in advance, so the
    API can compute them from station + date + hour alone."""
    df = df.copy()
    ts = pd.to_datetime(df["ts"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["year"] = ts.dt.year            # long-term growth of the system
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(ts.min().floor("D"), ts.max().ceil("D"))
    df["is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    # cyclic encoding: 23h and 0h are neighbours, not 23 apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["station"] = df["station"].astype(str).astype("category")
    return df


def load_demand(rebuild: bool = False) -> pd.DataFrame:
    """Build (or load the cached) hourly demand table, with features."""
    expected = {"station", "ts", TARGET, "lat", "lng"}
    if DEMAND_PATH.exists() and not rebuild:
        demand = pd.read_parquet(DEMAND_PATH)
        if not expected.issubset(demand.columns):      # cache made by an older version
            print("Cached demand table is outdated, rebuilding ...")
            rebuild = True
    if rebuild or not DEMAND_PATH.exists():
        demand = build_hourly_demand()
        demand.to_parquet(DEMAND_PATH, index=False)
    return add_time_features(demand)


def split_dates(df: pd.DataFrame, test_months: int = 12, valid_months: int = 12):
    """Test = the last 12 months in the data, validation = the 12 before that,
    train = everything earlier. Full years, so every split contains all seasons,
    and the split moves forward by itself when new months are downloaded."""
    last = pd.to_datetime(df["ts"]).max().to_period("M")
    test_start = (last - test_months + 1).to_timestamp()
    valid_start = (last - test_months - valid_months + 1).to_timestamp()
    return valid_start, test_start


def time_split(df: pd.DataFrame, test_months: int = 12, valid_months: int = 12):
    """Chronological three-way split: train < valid_start <= valid < test_start <= test.

    A random split would leak the future into training (neighbouring hours are
    almost identical), which makes the metrics look better than they really are.
    """
    valid_start, test_start = split_dates(df, test_months, valid_months)
    train = df[df["ts"] < valid_start]
    valid = df[(df["ts"] >= valid_start) & (df["ts"] < test_start)]
    test = df[df["ts"] >= test_start]
    return train, valid, test


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


RESULTS_PATH = Path("models/results.json")


def save_metrics(model_name: str, valid: dict, test: dict, where: str = "local",
                 note: str = "", path=RESULTS_PATH):
    """Store the VALIDATION and TEST scores of one model in a shared JSON file
    (read by the comparison notebook; every model notebook adds its own row).

    Rule from the course: models are compared and chosen on the VALIDATION
    columns. The test columns are only reported, once, and never used to choose.
    Choosing the winner on the test score would make the test score optimistic
    ("snoop bias").
    """
    import json
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    row = {"where": where, "note": note}
    row.update({f"valid_{k}": v for k, v in valid.items()})
    row.update({f"test_{k}": v for k, v in test.items()})
    allm = json.loads(path.read_text()) if path.exists() else {}
    allm[model_name] = row
    path.write_text(json.dumps(allm, indent=2))


def load_metrics(path=RESULTS_PATH) -> pd.DataFrame:
    """All saved models, best first - sorted on VALIDATION MAE, not on test."""
    import json
    rows = json.loads(Path(path).read_text())
    return pd.DataFrame(rows).T.sort_values("valid_MAE")


# ---- baselines ----------------------------------------------------------------
def seasonal_naive(df: pd.DataFrame, hours: int = 168) -> pd.Series:
    """Prediction = demand at the same station exactly `hours` earlier (168 h = the
    same hour one week ago). Needs the complete table, not a split: the first
    `hours` of every station come back as NaN. The standard time-series baseline."""
    d = df.sort_values(["station", "ts"])
    shifted = d.groupby("station", observed=True)[TARGET].shift(hours)
    return shifted.reindex(df.index)


def lookup_baseline(source: pd.DataFrame, target: pd.DataFrame,
                    keys=("station", "hour", "is_weekend")):
    """Prediction = mean demand in `source` for the same station x hour x weekend.
    Returns (predictions as an array aligned with `target`, the lookup table)."""
    keys = list(keys)
    lookup = source.groupby(keys, observed=True)[TARGET].mean().rename("pred")
    pred = target[keys].merge(lookup, left_on=keys, right_index=True, how="left")["pred"]
    return pred.fillna(source[TARGET].mean()).to_numpy(), lookup


# ---- station clusters (unsupervised learning, notebook 06) ----------------------
CLUSTER_PATH = Path("models/station_clusters.csv")


def station_profiles(df: pd.DataFrame, min_hours: int = 24 * 30) -> pd.DataFrame:
    """One row per station, 48 columns: the SHAPE of its demand over the day, for
    weekdays (wd_00 .. wd_23) and weekends (we_00 .. we_23). Each row is divided by
    its own total, so a quiet station and a busy station with the same daily rhythm
    look alike. Stations with fewer than `min_hours` rows are left out.
    Pass TRAINING rows only, otherwise test-period behaviour leaks into the clusters."""
    g = (df.groupby(["station", "is_weekend", "hour"], observed=True)[TARGET].mean()
           .unstack(["is_weekend", "hour"]).fillna(0))
    counts = df.groupby("station", observed=True).size()
    g = g[counts.reindex(g.index) >= min_hours]
    g = g.div(g.sum(axis=1), axis=0)
    g.columns = [f"{'we' if w else 'wd'}_{h:02d}" for w, h in g.columns]
    return g


def fit_station_clusters(profiles: pd.DataFrame, k: int, seed: int = 42):
    """Standardise the profiles, then KMeans. Returns (cluster per station, fitted objects).
    Scaling matters because KMeans uses distances; without it the columns with the
    largest values would decide the clusters."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(profiles)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(profiles))
    return pd.Series(km.labels_, index=profiles.index, name="cluster"), (scaler, km)


def add_cluster_feature(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Adds an integer column 'cluster'. Stations without a cluster (too new to have a
    profile) get -1, which the tree models treat as its own group."""
    df = df.copy()
    df["cluster"] = df["station"].astype(str).map(mapping).fillna(-1).astype(int)
    return df


def load_cluster_mapping(path=CLUSTER_PATH):
    """{station: cluster} from notebook 06, or None if 06 has not been run."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("station")["cluster"].to_dict()
