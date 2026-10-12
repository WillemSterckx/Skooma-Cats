"""Cleaning steps for the Citi Bike data (all years, 2013 - now).

Every cleaning step decided on in 02_clean_eda.ipynb is one function here, so
02 (exploring a sample), 03 (preparing everything) and the retraining pipeline
all run exactly the same code.

The data comes in two schemas:
  * legacy  (2013 - Jan 2021): tripduration, starttime, usertype (Subscriber/Customer), ...
  * current (Feb 2021 - now) : ride_id, rideable_type, started_at, member_casual (member/casual), ...
01_download maps both onto the CANONICAL column names below (values stay raw text);
set_types() then makes the values comparable.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("citibike_data/raw")
CLEAN_DIR = Path("citibike_data/clean")
SAMPLE_PATH = Path("citibike_data/raw_sample.parquet")

# ---- thresholds (justified in 02) -----------------------------------------
MIN_DURATION_MIN = 1            # shorter = false start / re-dock
MAX_DURATION_MIN = 24 * 60      # longer = lost, stolen or never docked
MAX_SPEED_KMH = 40              # e-bikes are capped around 32 km/h
LAT_RANGE = (40.45, 40.95)      # NYC + Jersey City
LNG_RANGE = (-74.30, -73.65)

# ---- 1. one schema for all years ------------------------------------------
CANONICAL = ["ride_id", "rideable_type", "started_at", "ended_at",
             "start_station_id", "start_station_name", "start_lat", "start_lng",
             "end_station_id", "end_station_name", "end_lat", "end_lng",
             "member_casual", "bike_id", "trip_duration", "birth_year", "gender"]

# Header names are compared after lower-casing and removing everything that is
# not a letter or digit, so "Start Station ID", "start station id" and
# "start_station_id" all become "startstationid".
_ALIASES = {
    # current schema
    "rideid": "ride_id", "rideabletype": "rideable_type",
    "startedat": "started_at", "endedat": "ended_at",
    "startstationid": "start_station_id", "startstationname": "start_station_name",
    "startlat": "start_lat", "startlng": "start_lng",
    "endstationid": "end_station_id", "endstationname": "end_station_name",
    "endlat": "end_lat", "endlng": "end_lng", "membercasual": "member_casual",
    # legacy schema
    "tripduration": "trip_duration", "starttime": "started_at", "stoptime": "ended_at",
    "startstationlatitude": "start_lat", "startstationlongitude": "start_lng",
    "endstationlatitude": "end_lat", "endstationlongitude": "end_lng",
    "bikeid": "bike_id", "usertype": "member_casual",
    "birthyear": "birth_year", "gender": "gender",
}


def canonical_name(header: str):
    """Map one CSV header (any year) to its canonical name, or None if unknown."""
    return _ALIASES.get(re.sub(r"[^a-z0-9]", "", header.lower()))


# ---- 2. types and labels ---------------------------------------------------
NA_STRINGS = ["", "NULL", "null", "\\N", "nan", "NaN"]
STATION_COLS = ["start_station_id", "start_station_name", "end_station_id", "end_station_name"]
COORD_COLS = ["start_lat", "start_lng", "end_lat", "end_lng"]
MEMBER_MAP = {"member": "member", "subscriber": "member",
              "casual": "casual", "customer": "casual"}
BIKE_MAP = {"classic_bike": "classic_bike", "docked_bike": "classic_bike",
            "electric_bike": "electric_bike"}
MEMBER_TYPES = ["member", "casual", "unknown"]
BIKE_TYPES = ["classic_bike", "electric_bike", "unknown"]


def parse_datetime(s: pd.Series) -> pd.Series:
    """The timestamp format changed over the years (2013-07-01 00:00:00,
    9/1/2014 00:00:04, 1/1/2015 0:01, 2024-01-01 00:00:00.123). One file always
    uses one format, so we try the fast fixed formats first."""
    for fmt in ("ISO8601", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s, format="mixed", errors="coerce")


def set_types(df: pd.DataFrame) -> pd.DataFrame:
    """Text placeholders -> NA, timestamps and coordinates -> real types,
    and the legacy labels translated to the current ones."""
    df = df.copy()
    for col in df.columns.intersection(CANONICAL):     # all raw columns are text
        df[col] = df[col].mask(df[col].isin(NA_STRINGS))
    df["started_at"] = parse_datetime(df["started_at"])
    df["ended_at"] = parse_datetime(df["ended_at"])
    for col in COORD_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["member_casual"] = df["member_casual"].str.strip().str.lower().map(MEMBER_MAP)
    # 2021-22 calls a classic bike in a dock "docked_bike": same bike, so we merge them
    df["rideable_type"] = df["rideable_type"].str.strip().str.lower().map(BIKE_MAP)
    return df


# ---- 3. duplicates ---------------------------------------------------------
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Current data: ride_id must be unique. Legacy data has no ride_id, so a
    duplicate there is the same bike with the same start, end and stations."""
    has_id = df["ride_id"].notna()
    current = df[has_id].drop_duplicates(subset="ride_id")
    legacy = df[~has_id].drop_duplicates(
        subset=["started_at", "ended_at", "start_station_name", "end_station_name", "bike_id"])
    return pd.concat([current, legacy]).sort_index()


# ---- 4. missing values -----------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Structural NAs are filled, NAs that make a trip unusable are dropped."""
    df = df.copy()
    df[STATION_COLS] = df[STATION_COLS].fillna("no_station")   # e-bike parked outside a dock
    df["rideable_type"] = df["rideable_type"].fillna("unknown")  # not recorded before 2021
    df["member_casual"] = df["member_casual"].fillna("unknown")
    return df.dropna(subset=["started_at", "ended_at"] + COORD_COLS)


# ---- 5. derived trip metrics + outliers ------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def add_trip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60
    df["distance_km"] = haversine_km(df.start_lat, df.start_lng, df.end_lat, df.end_lng)
    hours = df["duration_min"].where(df["duration_min"] > 0) / 60
    df["speed_kmh"] = df["distance_km"] / hours
    return df


def filter_duration(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN)]


def filter_location(df: pd.DataFrame) -> pd.DataFrame:
    ok = (df.start_lat.between(*LAT_RANGE) & df.end_lat.between(*LAT_RANGE)
          & df.start_lng.between(*LNG_RANGE) & df.end_lng.between(*LNG_RANGE))
    return df[ok]


def filter_speed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["speed_kmh"] <= MAX_SPEED_KMH]


# ---- 6. final columns and categoricals --------------------------------------
CLEAN_COLUMNS = ["started_at", "ended_at", *STATION_COLS, *COORD_COLS,
                 "rideable_type", "member_casual", "duration_min", "distance_km"]


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only what later steps need. Dropped: ride_id (only for duplicates),
    bike_id / trip_duration (replaced by duration_min), birth_year / gender
    (only exist before 2021, and personal data). Fixed category lists keep
    every output file's schema identical."""
    df = df[CLEAN_COLUMNS].copy()
    df["started_at"] = df["started_at"].astype("datetime64[us]")
    df["ended_at"] = df["ended_at"].astype("datetime64[us]")
    df["member_casual"] = pd.Categorical(df["member_casual"], categories=MEMBER_TYPES)
    df["rideable_type"] = pd.Categorical(df["rideable_type"], categories=BIKE_TYPES)
    return df.reset_index(drop=True)


STEPS = [set_types, drop_duplicates, handle_missing, add_trip_metrics,
         filter_duration, filter_location, filter_speed, finalize]


def clean(df: pd.DataFrame):
    """Run every step. Returns the clean frame and {step: rows left}."""
    report = {"raw": len(df)}
    for step in STEPS:
        df = step(df)
        report[step.__name__] = len(df)
    return df, report


# ---- helpers for 02 and 03 -------------------------------------------------
def load_raw_sample(pct: float = 0.5, raw_dir: Path = RAW_DIR,
                    cache: Path = SAMPLE_PATH, rebuild: bool = False) -> pd.DataFrame:
    """Uniform random sample of ALL raw trips (every year), for EDA. DuckDB
    reads the parquet files without loading everything into memory."""
    import duckdb
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    con = duckdb.connect()
    sample = con.execute(f"""
        SELECT * FROM read_parquet('{raw_dir.as_posix()}/**/*.parquet', union_by_name=true)
        USING SAMPLE {pct} PERCENT (bernoulli, 42)""").df()
    sample.to_parquet(cache, index=False)
    return sample


def prepare_all(raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR,
                rebuild: bool = False, batch_size: int = 1_000_000) -> pd.DataFrame:
    """Clean every raw parquet file in batches (memory stays ~1M rows) and write
    one clean file per raw file. Finished files are skipped, so it can resume.
    Returns the cleaning report (rows left after each step, per file)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    report_path = clean_dir / "_report.csv"
    if rebuild and report_path.exists():
        report_path.unlink()
    files = sorted(raw_dir.rglob("*.parquet"))
    for i, f in enumerate(files, 1):
        out = clean_dir / f.relative_to(raw_dir)
        if out.exists() and not rebuild:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        writer, totals = None, {}
        for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_size):
            df, rep = clean(batch.to_pandas())
            table = pa.Table.from_pandas(df, preserve_index=False)
            writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            totals = {k: totals.get(k, 0) + v for k, v in rep.items()}
        if writer:
            writer.close()
            tmp.replace(out)
        row = pd.DataFrame([{"file": f.relative_to(raw_dir).as_posix(), **totals}])
        row.to_csv(report_path, mode="a", header=not report_path.exists(), index=False)
        print(f"[{i}/{len(files)}] {f.name}: {totals.get('raw', 0):,} -> {totals.get('finalize', 0):,} rows")
    return pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
