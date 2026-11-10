"""Download the CARE to Compare wind turbine SCADA dataset and convert it to Parquet.

Output layout (under data/):
    <farm>/event_info.parquet
    <farm>/feature_description.parquet
    <farm>/datasets/<event_id>.parquet

The Kaggle upload contains two versions of every file: the original semicolon-separated
files (newer Zenodo release, event_info includes an `asset` column) and `comma_*` copies
of an older release with different timestamps and values. Only the semicolon files are
used here so data and event metadata stay consistent. Files are downloaded one by one
rather than via a full dataset download, which would also fetch the `comma_*` duplicates
(~40 GB total vs ~20 GB).

Usage:
    python data_process.py                 # all farms (Farm C alone is ~17 GB of CSV)
    python data_process.py --farms A B     # selected farms
    python data_process.py --overwrite     # rebuild existing parquet files
    python data_process.py --delete-csv    # remove each cached CSV after converting it
"""

import argparse
from pathlib import Path

import kagglehub
import pandas as pd

HANDLE = "azizkasimov/wind-turbine-scada-data-for-early-fault-detection"
FARMS = ("A", "B", "C")
DATA_DIR = Path(__file__).resolve().parent / "data"
SEP = ";"

META_COLUMNS = {
    "time_stamp": None,  # parsed to datetime separately
    "asset_id": "int16",
    "id": "int32",
    "train_test": "category",
    "status_type_id": "int8",
}


def download(path: str) -> Path:
    return Path(kagglehub.dataset_download(HANDLE, path=path))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=SEP, engine="pyarrow")


def convert_metadata(farm: str, out_dir: Path) -> pd.DataFrame:
    event_info = read_csv(download(f"Wind Farm {farm}/event_info.csv"))
    for col in ("event_start", "event_end"):
        event_info[col] = pd.to_datetime(event_info[col])
    event_info["event_label"] = event_info["event_label"].astype("category")
    event_info.to_parquet(out_dir / "event_info.parquet", index=False)

    features = read_csv(download(f"Wind Farm {farm}/feature_description.csv"))
    features.to_parquet(out_dir / "feature_description.parquet", index=False)
    return event_info


def convert_event(farm: str, event_id: int, out_path: Path, delete_csv: bool) -> None:
    csv_path = download(f"Wind Farm {farm}/datasets/{event_id}.csv")
    df = read_csv(csv_path)

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    for col, dtype in META_COLUMNS.items():
        if dtype is not None:
            df[col] = df[col].astype(dtype)
    # Sensor values as float32: halves storage and matches PyTorch's default dtype.
    sensor_cols = [c for c in df.columns if c not in META_COLUMNS]
    df[sensor_cols] = df[sensor_cols].astype("float32")

    df.to_parquet(out_path, index=False, compression="zstd")
    if delete_csv:
        csv_path.unlink()


def process_farm(farm: str, overwrite: bool, delete_csv: bool) -> None:
    out_dir = DATA_DIR / farm
    (out_dir / "datasets").mkdir(parents=True, exist_ok=True)

    event_info = convert_metadata(farm, out_dir)
    event_ids = sorted(event_info["event_id"].tolist())
    print(f"Wind Farm {farm}: {len(event_ids)} datasets")

    for i, event_id in enumerate(event_ids, 1):
        out_path = out_dir / "datasets" / f"{event_id}.parquet"
        if out_path.exists() and not overwrite:
            print(f"  [{i}/{len(event_ids)}] {event_id}: exists, skipping")
            continue
        print(f"  [{i}/{len(event_ids)}] {event_id}: converting")
        convert_event(farm, event_id, out_path, delete_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--farms", nargs="+", choices=FARMS, default=list(FARMS))
    parser.add_argument("--overwrite", action="store_true", help="rebuild parquet files that already exist")
    parser.add_argument("--delete-csv", action="store_true", help="delete cached CSVs after conversion to save disk")
    args = parser.parse_args()

    for farm in args.farms:
        process_farm(farm, args.overwrite, args.delete_csv)


if __name__ == "__main__":
    main()
