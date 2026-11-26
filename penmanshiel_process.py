"""Convert the Penmanshiel wind farm SCADA export (Greenbyte CSV) to Parquet.

Penmanshiel is a 14-turbine Senvion MM82 (2.05 MW) farm in Scotland, released as open data by Cubico.
There is no turbine 03. The raw download lives in data/Penmanshiel/ as one folder per year (and per turbine
group, WT01-10 / WT11-15). Each folder holds one `Turbine_Data_*.csv` (10-minute SCADA) and one `Status_*.csv`
(event log) per turbine.

Output layout (under data/Penmanshiel/):
    turbines.parquet               static metadata per turbine (Penmanshiel_WT_static.csv)
    feature_description.parquet    one row per SCADA column: column, original label, name, stat, unit
    scada/<turbine_id>.parquet     10-minute SCADA, all years concatenated, one row per time_stamp
    status/<turbine_id>.parquet    status/event log, all years concatenated

`turbine_id` is the turbine number (1, 2, 4, ..., 15). SCADA columns are renamed to snake_case. A label
`<name>, <stat> (<unit>)` becomes `<name>_<stat>` with stat in {min, max, std}, so `Wind speed, Standard
deviation (m/s)` -> `wind_speed_std`. The original label and unit are kept in feature_description.

The yearly files 2016-2020 share one 300-column header. The 2023-02 files have 372 columns: 85 extra signals,
APE-2 dropped, and the acceleration unit renamed from `mm/ss` to `mm/s2` (treated as the same column). Each
turbine's file holds the union of columns, with NaN where a period didn't record a signal. All signals are
stored as float64, because which columns are integer differs between files (an integer column with a gap
is read as float).

Data quirks handled here:
- The 2023 SCADA exports repeat timestamps (2023-02 WT01-10 up to 8x, 2023 WT11-15 ~47x): one full row, the
  rest only a few availability columns, never conflicting. Rows are merged per timestamp, reading in chunks.
- Status files contain exact duplicate rows (dropped). From 2018 on, most status rows are point-in-time
  messages with no end time or duration (NaT).
- WT01-10 have no data for January 2023.

Usage:
    python penmanshiel_process.py                  # all turbines
    python penmanshiel_process.py --turbines 1 2   # selected turbines
    python penmanshiel_process.py --overwrite      # rebuild existing parquet files
"""

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "data" / "Penmanshiel"
OUT_DIR = RAW_DIR
TIME_COL = "Date and time"

# Parenthesised text that counts as a unit; anything else in brackets, e.g. "(Contractual Global)", is part
# of the signal name.
UNITS = {"m/s", "°", "°C", "kW", "kWh", "kvar", "kvarh", "kVA", "A", "V", "Hz", "RPM", "Nm", "bar", "%", "s",
         "h", "mm/s2", "cosphi"}
UNIT_ALIASES = {"mm/ss": "mm/s2"}
STATS = {"standard deviation": "std", "stddev": "std", "stdev": "std", "std": "std",
         "minimum": "min", "min": "min", "maximum": "max", "max": "max"}
UNIT_SLUG = {"%": "pct", "°": "deg", "°C": "degc", "mm/s2": "mm_s2", "m/s": "m_s"}

FILE_RE = re.compile(r"^(Turbine_Data|Status)_Penmanshiel_(\d+)_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2})_\d+\.csv$")


# ---------------------------------------------------------------------------------------------------------
# Reading the Greenbyte CSVs
# ---------------------------------------------------------------------------------------------------------

def header_row(path: Path) -> int:
    """Line index of the column header: the line just before the first data line.

    Greenbyte files start with `#` comment lines of varying count; in turbine data files the header line itself
    also starts with `# ` (`# Date and time,...`), so the header is found as the line before the first line
    that begins with a date.
    """
    with open(path, encoding="utf-8-sig") as f:
        prev = None
        for i, line in enumerate(f):
            if re.match(r"\d{4}-\d{2}-\d{2}", line):
                return i - 1
            if not line.startswith("#") or "," in line:
                prev = i
        if prev is None:
            raise ValueError(f"no header found in {path}")
        return prev  # header but no data rows


def read_greenbyte(path: Path) -> pd.DataFrame:
    # Status files write "-" for a missing end/duration: point-in-time messages (common from 2018 on, e.g.
    # "Brake program 50") and events still open at export time.
    df = pd.read_csv(path, skiprows=header_row(path), encoding="utf-8-sig", na_values=["NaN", "-"],
                     low_memory=False)
    df.columns = [c.lstrip("# ").strip() for c in df.columns]
    return df


def find_files(kind: str) -> dict[int, list[Path]]:
    """{turbine_id: [paths]} for kind "Turbine_Data" or "Status", sorted by period start."""
    found: dict[int, list[tuple[str, Path]]] = {}
    for path in RAW_DIR.glob("Penmanshiel_SCADA_*/*.csv"):
        m = FILE_RE.match(path.name)
        if m and m.group(1) == kind:
            found.setdefault(int(m.group(2)), []).append((m.group(3), path))
    return {t: [p for _, p in sorted(v)] for t, v in sorted(found.items())}


# ---------------------------------------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------------------------------------

def parse_label(label: str) -> dict:
    """Split a Greenbyte label like 'Wind speed, Standard deviation (m/s)' into name, stat and unit."""
    text, unit = label, None
    for m in reversed(list(re.finditer(r"\s*\(([^()]*)\)", text))):
        candidate = UNIT_ALIASES.get(m.group(1), m.group(1))
        if candidate in UNITS:
            unit = candidate
            text = text[:m.start()] + text[m.end():]
            break
    parts = [p.strip() for p in text.split(",")]
    name, stat = parts[0], None
    for extra in parts[1:]:
        low = extra.lower()
        key = next((k for k in sorted(STATS, key=len, reverse=True) if low == k or low.startswith(k + " ")), None)
        if key is not None and stat is None:
            stat = STATS[key]
            rest = low[len(key):].strip()  # e.g. "Min counter" -> stat min, "counter" stays in the name
            if rest:
                name = f"{name} {rest}"
        else:
            name = f"{name} {extra}"
    return {"label": label, "name": name, "stat": stat, "unit": unit}


def snake(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")


def build_feature_description(labels: list[str]) -> pd.DataFrame:
    """One row per original label with its snake_case column name. Labels that differ only in the
    mm/ss -> mm/s2 unit spelling map to the same column."""
    rows = [parse_label(lab) for lab in labels]
    for r in rows:
        r["column"] = snake(r["name"]) + (f"_{r['stat']}" if r["stat"] else "")
    # Disambiguate genuinely different signals that end up with the same name (same name, different unit).
    key = lambda r: (r["name"], r["stat"], r["unit"])
    by_col: dict[str, set] = {}
    for r in rows:
        by_col.setdefault(r["column"], set()).add(key(r))
    for r in rows:
        if len(by_col[r["column"]]) > 1 and r["unit"]:
            base = snake(r["name"]) + f"_{UNIT_SLUG.get(r['unit'], snake(r['unit']))}"
            r["column"] = base + (f"_{r['stat']}" if r["stat"] else "")
    fd = pd.DataFrame(rows)[["column", "label", "name", "stat", "unit"]]
    clash = fd.drop_duplicates(["column", "name", "stat", "unit"])["column"]
    if clash.duplicated().any():
        raise ValueError(f"column name clash: {sorted(clash[clash.duplicated()])}")
    return fd


# ---------------------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------------------

def merge_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per time_stamp, sorted; repeated rows merged by taking the first non-null value per column.

    The 2023 exports repeat timestamps: 2023-02 (WT01-10) up to 8 times, 2023 WT11-15 about 47 times. Each
    timestamp has one full row plus near-empty rows carrying a few availability columns. Their non-null values
    never conflict, so merging loses nothing.
    """
    if not df["time_stamp"].duplicated().any():
        return df.sort_values("time_stamp", ignore_index=True)
    with warnings.catch_warnings():  # pandas warns about fragmentation from inside groupby().first()
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return df.groupby("time_stamp", sort=True).first().reset_index()


def read_scada_file(path: Path, label_to_col: dict[str, str], columns: list[str],
                    chunksize: int = 200_000) -> tuple[pd.DataFrame, int]:
    """One SCADA CSV as (frame with time_stamp + `columns` as float64, raw row count).

    Read in chunks and merged per chunk, so the heavily repeated 2023 WT11-15 files (~2.5M rows, ~7 GB if read
    whole) stay small in memory.
    """
    parts, n_raw = [], 0
    reader = pd.read_csv(path, skiprows=header_row(path), encoding="utf-8-sig", na_values=["NaN", "-"],
                         low_memory=False, chunksize=chunksize)
    for chunk in reader:
        chunk.columns = [c.lstrip("# ").strip() for c in chunk.columns]
        n_raw += len(chunk)
        chunk = chunk.rename(columns={c: label_to_col[c] for c in chunk.columns if c != TIME_COL})
        chunk = pd.concat([pd.to_datetime(chunk[TIME_COL]).rename("time_stamp"),
                           chunk.reindex(columns=columns).astype("float64")], axis=1)
        parts.append(merge_duplicate_timestamps(chunk))
    return merge_duplicate_timestamps(pd.concat(parts, ignore_index=True)), n_raw


def convert_scada(turbine: int, paths: list[Path], label_to_col: dict[str, str], columns: list[str],
                  out_path: Path) -> None:
    frames, n = [], 0
    for path in paths:
        df, n_raw = read_scada_file(path, label_to_col, columns)
        frames.append(df)
        n += n_raw
    df = merge_duplicate_timestamps(pd.concat(frames, ignore_index=True))
    df = pd.concat([df[["time_stamp"]], pd.Series(turbine, index=df.index, dtype="int8", name="turbine_id"),
                    df[columns]], axis=1)
    df.to_parquet(out_path, index=False)
    print(f"  scada  T{turbine:02d}: {len(df):,} rows ({n - len(df):,} duplicate-timestamp rows merged), "
          f"{df['time_stamp'].min():%Y-%m-%d} to {df['time_stamp'].max():%Y-%m-%d}")


def convert_status(turbine: int, paths: list[Path], out_path: Path) -> None:
    df = pd.concat([read_greenbyte(p) for p in paths], ignore_index=True)
    df = df.rename(columns=lambda c: snake(c))
    for col in ("timestamp_start", "timestamp_end"):
        df[col] = pd.to_datetime(df[col])
    df["duration"] = pd.to_timedelta(df["duration"].map(_hms_to_timedelta_str))
    n = len(df)
    df = df.drop_duplicates().sort_values(["timestamp_start", "timestamp_end"], ignore_index=True)
    # Text columns: low-cardinality ones as category, the rest as string. The 2023-02 export adds global/custom
    # contract category columns (NaN in earlier periods); an all-empty one would otherwise be read as float.
    for col in df.columns.drop(["timestamp_start", "timestamp_end", "duration", "code"]):
        df[col] = df[col].astype("string")
        if col != "message" and df[col].nunique() < 200:
            df[col] = df[col].astype("category")
    df["code"] = df["code"].astype("int64")
    df.insert(0, "turbine_id", pd.Series(turbine, index=df.index, dtype="int8"))
    df.to_parquet(out_path, index=False)
    print(f"  status T{turbine:02d}: {len(df):,} events ({n - len(df):,} exact duplicates dropped)")


def _hms_to_timedelta_str(value) -> str | None:
    """'94:47:52' (hours may exceed 24) -> '94h47m52s'."""
    if pd.isna(value):
        return None
    h, m, s = str(value).split(":")
    return f"{h}h{m}m{s}s"


def convert_static(out_path: Path) -> None:
    df = pd.read_csv(RAW_DIR / "Penmanshiel_WT_static.csv", encoding="utf-8-sig").dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1).rename(columns=snake)
    df.insert(0, "turbine_id", df["title"].str.extract(r"(\d+)$")[0].astype("int8"))
    df["commercial_operations_date"] = pd.to_datetime(df["commercial_operations_date"], dayfirst=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turbines", type=int, nargs="+", help="turbine numbers to convert (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing parquet files")
    args = parser.parse_args()

    scada_files, status_files = find_files("Turbine_Data"), find_files("Status")
    if not scada_files:
        raise SystemExit(f"No Turbine_Data_*.csv files found under {RAW_DIR}/Penmanshiel_SCADA_*")
    turbines = args.turbines or sorted(scada_files)
    if missing := set(turbines) - set(scada_files):
        raise SystemExit(f"No SCADA files for turbine(s) {sorted(missing)}; available: {sorted(scada_files)}")

    (OUT_DIR / "scada").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "status").mkdir(parents=True, exist_ok=True)
    convert_static(OUT_DIR / "turbines.parquet")

    # Column names come from the union of all headers, so every turbine's file has the same columns.
    labels = list(dict.fromkeys(c for paths in scada_files.values() for p in paths
                                for c in pd.read_csv(p, skiprows=header_row(p), nrows=0,
                                                     encoding="utf-8-sig").columns.str.lstrip("# ").str.strip()
                                if c != TIME_COL))
    fd = build_feature_description(labels)
    fd.to_parquet(OUT_DIR / "feature_description.parquet", index=False)
    label_to_col = dict(zip(fd["label"], fd["column"]))
    columns = list(dict.fromkeys(fd["column"]))
    print(f"{len(labels)} distinct labels -> {len(columns)} columns; converting turbines {turbines}")

    for t in turbines:
        scada_out = OUT_DIR / "scada" / f"{t}.parquet"
        if args.overwrite or not scada_out.exists():
            convert_scada(t, scada_files[t], label_to_col, columns, scada_out)
        else:
            print(f"  scada  T{t:02d}: exists, skipping")
        status_out = OUT_DIR / "status" / f"{t}.parquet"
        if t in status_files and (args.overwrite or not status_out.exists()):
            convert_status(t, status_files[t], status_out)


if __name__ == "__main__":
    main()
