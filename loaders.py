"""Load the Parquet files produced by data_process.py.

Each dataset (one per event_id) is one turbine's time series: ~1 year of `train` rows
followed by a `prediction` period containing exactly one event. For anomaly events the
rows between event_start_id and event_end_id (inclusive) are the labelled fault window.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
FARMS = ("A", "B", "C")
META_COLUMNS = ("time_stamp", "asset_id", "id", "train_test", "status_type_id")
# status_type_id 0 (normal operation) and 2 (idling) are considered normal behaviour.
NORMAL_STATUS = (0, 2)


def _farm_dir(farm: str) -> Path:
    path = DATA_DIR / farm
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python data_process.py --farms {farm}` first")
    return path


def load_event_info(farm: str | None = None) -> pd.DataFrame:
    """Event metadata for one farm, or all converted farms (with a `farm` column) if None."""
    if farm is not None:
        return pd.read_parquet(_farm_dir(farm) / "event_info.parquet")
    frames = [
        load_event_info(f).assign(farm=f) for f in FARMS if (DATA_DIR / f / "event_info.parquet").exists()
    ]
    if not frames:
        raise FileNotFoundError(f"No converted farms found in {DATA_DIR}; run data_process.py first")
    return pd.concat(frames, ignore_index=True)


def load_feature_description(farm: str) -> pd.DataFrame:
    return pd.read_parquet(_farm_dir(farm) / "feature_description.parquet")


def list_event_ids(farm: str, label: str | None = None) -> list[int]:
    """Event ids for a farm, optionally filtered by label ('anomaly' or 'normal')."""
    info = load_event_info(farm)
    if label is not None:
        info = info[info["event_label"] == label]
    return sorted(info["event_id"].tolist())


def load_event(farm: str, event_id: int, columns: list[str] | None = None, add_labels: bool = True) -> pd.DataFrame:
    """Load one dataset.

    columns: subset of sensor columns to read (meta columns are always included).
    add_labels: add `is_anomaly` (row is inside the event window of an anomaly event)
        and `is_normal_status` (status_type_id in NORMAL_STATUS) columns.
    """
    read_cols = None if columns is None else list(dict.fromkeys([*META_COLUMNS, *columns]))
    df = pd.read_parquet(_farm_dir(farm) / "datasets" / f"{event_id}.parquet", columns=read_cols)

    if add_labels:
        info = load_event_info(farm).set_index("event_id").loc[event_id]
        in_window = df["id"].between(info["event_start_id"], info["event_end_id"])
        df["is_anomaly"] = in_window & (info["event_label"] == "anomaly")
        df["is_normal_status"] = df["status_type_id"].isin(NORMAL_STATUS)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into its train and prediction parts."""
    is_train = df["train_test"] == "train"
    return df[is_train], df[~is_train]


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Sensor feature columns (everything except meta and label columns)."""
    exclude = {*META_COLUMNS, "is_anomaly", "is_normal_status"}
    return [c for c in df.columns if c not in exclude]
