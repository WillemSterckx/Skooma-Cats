"""Download, unpack and assemble ALL NYC Citi Bike trip data (2013 - now).

* The archive list is read from the S3 bucket itself, so new months are picked
  up automatically (useful for the retraining pipeline).
* Jersey City files (JC-...) are skipped: this project is about NYC.
* 2013-2023 come as one zip per year (sometimes with zips inside),
  2024 onwards as one zip per month. Both are handled.
* Each CSV is converted to parquet with ONE common set of column names
  (see src/cleaning.CANONICAL). Values are kept as raw text: cleaning is 03's job.
* Finished archives are listed in raw/_done.txt, so a crash or a re-run only
  downloads what is missing.
"""
import csv
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from src.cleaning import CANONICAL, canonical_name

BASE_URL = "https://s3.amazonaws.com/tripdata/"
DATA_DIR = Path("citibike_data")
RAW_DIR = DATA_DIR / "raw"
TMP_DIR = DATA_DIR / "tmp"
DONE_FILE = RAW_DIR / "_done.txt"
NYC_ARCHIVE = re.compile(r"^(\d{4}|\d{6})-citibike-tripdata(\.csv)?\.zip$")


def list_archives(base_url: str = BASE_URL) -> list[str]:
    """All NYC archive names in the bucket, oldest first."""
    keys, marker = [], ""
    while True:
        url = base_url + (f"?marker={urllib.parse.quote(marker)}" if marker else "")
        with urllib.request.urlopen(url) as r:
            root = ET.fromstring(r.read())
        page = [el.text for el in root.iter() if el.tag.split("}")[-1] == "Key"]
        keys += page
        truncated = [el.text for el in root.iter() if el.tag.split("}")[-1] == "IsTruncated"]
        if not page or truncated[:1] != ["true"]:
            break
        marker = page[-1]
    return sorted(k for k in keys if NYC_ARCHIVE.match(k))


def _download(key: str, base_url: str) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    target = TMP_DIR / key
    part = target.with_suffix(".part")          # only renamed when complete
    print(f"Downloading {key} ...", flush=True)
    urllib.request.urlretrieve(base_url + key, part)
    part.replace(target)
    return target


def _column_names(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8-sig", errors="replace") as f:
        header = next(csv.reader(f))
    names = []
    for i, h in enumerate(header):
        c = canonical_name(h)
        names.append(c if c and c not in names else f"_unused_{i}")
    ignored = [h for h, n in zip(header, names) if n.startswith("_unused_")]
    if ignored:
        print(f"    ignoring unknown columns {ignored}")
    return names


def csv_to_parquet(csv_path: Path, out_path: Path) -> None:
    """One CSV (any year) -> parquet with the canonical columns, all as text."""
    names = _column_names(csv_path)
    tmp = out_path.with_suffix(".tmp")
    cols = ", ".join(f'"{c}"' if c in names else f"NULL::VARCHAR AS {c}" for c in CANONICAL)
    try:
        duckdb.connect().execute(f"""
            COPY (SELECT {cols}, '{csv_path.name}' AS source_file
                  FROM read_csv('{csv_path.as_posix()}', header=false, skip=1,
                                names={names}, all_varchar=true, delim=',', quote='"'))
            TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""")
    except duckdb.Error as e:
        # fallback for files DuckDB cannot parse (encoding problems, ragged rows)
        print(f"    DuckDB failed on {csv_path.name} ({type(e).__name__}), using pandas")
        tmp.unlink(missing_ok=True)
        import pyarrow as pa
        import pyarrow.parquet as pq
        writer = None
        for chunk in pd.read_csv(csv_path, dtype=str, header=0, names=names, chunksize=1_000_000,
                                 keep_default_na=False, encoding_errors="replace",
                                 on_bad_lines="warn"):
            frame = pd.DataFrame({c: chunk[c] if c in chunk else None for c in CANONICAL},
                                 dtype="string").assign(source_file=csv_path.name)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            writer = writer or pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
        if writer:
            writer.close()
    tmp.replace(out_path)


def _convert_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract one member at a time (keeps disk use low), recurse into inner zips."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            low = member.lower()
            if "__macosx" in low or member.endswith("/"):
                continue
            if low.endswith(".csv"):
                extracted = Path(z.extract(member, TMP_DIR))
                print(f"  converting {Path(member).name}", flush=True)
                csv_to_parquet(extracted, out_dir / (Path(member).stem + ".parquet"))
                extracted.unlink()
            elif low.endswith(".zip"):
                inner = Path(z.extract(member, TMP_DIR))
                _convert_zip(inner, out_dir)
                inner.unlink()


def download_all(years=None, base_url: str = BASE_URL) -> list[str]:
    """Download and convert every archive not done yet.
    years: optional list such as [2019, 2024] to test on a subset first."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    archives = list_archives(base_url)
    if years:
        archives = [a for a in archives if int(a[:4]) in set(years)]
    todo = [a for a in archives if a not in done]
    print(f"{len(archives)} archives, {len(archives) - len(todo)} already done, {len(todo)} to go")
    for key in todo:
        zip_path = _download(key, base_url)
        _convert_zip(zip_path, RAW_DIR / key.split("-")[0])   # raw/2019/..., raw/202401/...
        zip_path.unlink()
        with open(DONE_FILE, "a") as f:
            f.write(key + "\n")
    return archives
