"""Penmanshiel data access, event flags and healthy-operation mask.

Reads the Parquet files written by `penmanshiel_process.py` (data/Penmanshiel/). Flags are kept as separate
boolean columns so an analysis can choose which ones to exclude; `healthy_mask` combines a chosen set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "Penmanshiel"
RATED_POWER = 2050.0
COMMERCIAL_OPERATION = pd.Timestamp("2016-09-01")
SLOT = pd.Timedelta(minutes=10)

# Plain-English labels for the signals used in the analysis.
SIGNAL_LABELS = {
    "wind_speed": "Wind speed",
    "power": "Active power",
    "rotor_speed": "Rotor speed",
    "generator_rpm": "Generator speed",
    "blade_angle_pitch_position_a": "Blade pitch angle",
    "nacelle_ambient_temperature": "Ambient temperature",
    "nacelle_temperature": "Nacelle temperature",
    "front_bearing_temperature": "Gearbox bearing 1 (axle bearing 1)",
    "rear_bearing_temperature": "Gearbox bearing 2 (axle bearing 2)",
    "gear_oil_temperature": "Gearbox oil (sump)",
    "gear_oil_inlet_temperature": "Gearbox oil inlet",
    "generator_bearing_front_temperature": "Generator bearing, drive end",
    "generator_bearing_rear_temperature": "Generator bearing, non-drive end",
    "stator_temperature_1": "Generator stator winding",
    "rotor_bearing_temp": "Main (rotor) bearing",
    "turbine_power_setpoint": "Power setpoint",
}

# Physically plausible ranges; values outside are treated as sensor errors.
PLAUSIBLE = {
    "wind_speed": (0, 40),
    "power": (-50, RATED_POWER * 1.1),
    "rotor_speed": (0, 20),
    "generator_rpm": (0, 2000),
    "blade_angle_pitch_position_a": (-5, 95),
    "nacelle_ambient_temperature": (-25, 40),
    "nacelle_temperature": (-20, 50),
    "front_bearing_temperature": (-20, 110),
    "rear_bearing_temperature": (-20, 110),
    "gear_oil_temperature": (-20, 90),
    "gear_oil_inlet_temperature": (-20, 90),
    "generator_bearing_front_temperature": (-20, 110),
    "generator_bearing_rear_temperature": (-20, 110),
    "stator_temperature_1": (-20, 160),
    "rotor_bearing_temp": (-20, 90),
}


# ---------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------

def load_turbines() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "turbines.parquet")


def load_scada(turbines, columns, start=None, end=None, dtype="float32") -> pd.DataFrame:
    """Long frame (turbine_id, time_stamp, *columns) for the given turbines, sorted by turbine then time."""
    filters = []
    if start is not None:
        filters.append(("time_stamp", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("time_stamp", "<", pd.Timestamp(end)))
    frames = []
    for t in turbines:
        d = pd.read_parquet(DATA_DIR / "scada" / f"{t}.parquet", columns=["time_stamp", *columns],
                            filters=filters or None)
        d[columns] = d[columns].astype(dtype)
        d.insert(0, "turbine_id", np.int8(t))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_status(turbines) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(DATA_DIR / "status" / f"{t}.parquet") for t in turbines],
                     ignore_index=True)


# ---------------------------------------------------------------------------------------------------------
# Flags from the status log
# ---------------------------------------------------------------------------------------------------------

# Warning codes that reduce output (treated as a derate rather than an alarm).
POWER_REDUCED_CODES = {9000, 3220}  # P output externally reduced; Reduced power converter
# Warning codes unrelated to drivetrain condition or output: obstacle light, clock sync, comms, control-box
# heating/fans, pitch battery charge cycles. They can last weeks, so excluding them costs a lot of data.
BENIGN_WARNING_CODES = {5000, 7324, 8400, 7057, 4607, 716, 717, 718}

# Each flag is a rule on timed status rows (rows with an end time). A 10-minute slot gets the flag if any
# matching event overlaps it.
STATUS_FLAG_RULES = {
    "stop": lambda s: s["status"] == "Stop",
    "warning": lambda s: (s["status"] == "Warning") & ~s["code"].isin(POWER_REDUCED_CODES | BENIGN_WARNING_CODES),
    "warning_benign": lambda s: (s["status"] == "Warning") & s["code"].isin(BENIGN_WARNING_CODES),
    "power_reduced": lambda s: (s["status"] == "Warning") & s["code"].isin(POWER_REDUCED_CODES),
    "curtailment": lambda s: s["status"] == "Curtailment",
    "communication": lambda s: s["status"] == "Communication",
    "forced_outage": lambda s: s["iec_category"] == "Forced outage",
    "maintenance": lambda s: (s["iec_category"] == "Scheduled Maintenance")
                             | (s["service_contract_category"] == "Manual stop (service)  (9)"),
    "derate": lambda s: s["iec_category"] == "Partial Performance",
    "requested_shutdown": lambda s: s["iec_category"] == "Requested Shutdown",
    "technical_standby": lambda s: s["iec_category"] == "Technical Standby",
    "electrical_spec": lambda s: s["iec_category"] == "Out of Electrical Specification",
}


def _mark_intervals(times: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Boolean per slot [t, t+10 min): True if it overlaps any interval [start, end)."""
    slot_end = times + SLOT.to_timedelta64()
    i0 = np.searchsorted(slot_end, starts, side="right")  # first slot ending after the start
    i1 = np.searchsorted(times, ends, side="left")        # first slot starting at/after the end
    diff = np.zeros(len(times) + 1, dtype=np.int32)
    keep = i1 > i0
    np.add.at(diff, i0[keep], 1)
    np.add.at(diff, i1[keep], -1)
    return np.cumsum(diff[:-1]) > 0


def status_flags(scada: pd.DataFrame, status: pd.DataFrame, rules=None, post_event: pd.Timedelta = None,
                 post_event_flags=("stop", "forced_outage", "maintenance", "requested_shutdown")) -> pd.DataFrame:
    """One boolean column per rule, aligned with `scada` rows (needs turbine_id, time_stamp; sorted per turbine).

    `post_event` extends the listed flags by this long after each event ends (as `post_event`), because
    component temperatures take time to settle after a stop.
    """
    rules = rules or STATUS_FLAG_RULES
    timed = status[status["timestamp_end"].notna()]
    out = {name: np.zeros(len(scada), dtype=bool) for name in rules}
    if post_event is not None:
        out["post_event"] = np.zeros(len(scada), dtype=bool)
    for t, rows in scada.groupby("turbine_id", sort=False).indices.items():
        times = scada["time_stamp"].to_numpy()[rows]
        ev = timed[timed["turbine_id"] == t]
        for name, rule in rules.items():
            e = ev[rule(ev).fillna(False).to_numpy(bool)]
            out[name][rows] = _mark_intervals(times, e["timestamp_start"].to_numpy(), e["timestamp_end"].to_numpy())
        if post_event is not None:
            e = ev[np.any([rules[f](ev).fillna(False).to_numpy(bool) for f in post_event_flags], axis=0)]
            ends = e["timestamp_end"].to_numpy()
            out["post_event"][rows] = _mark_intervals(times, ends, ends + post_event.to_timedelta64())
    return pd.DataFrame(out, index=scada.index)


# ---------------------------------------------------------------------------------------------------------
# Flags from the SCADA values
# ---------------------------------------------------------------------------------------------------------

def scada_flags(scada: pd.DataFrame, required, min_power=20.0, min_rotor_speed=5.0,
                derate_wind_speed=11.0, derate_setpoint_fraction=0.95) -> pd.DataFrame:
    """SCADA-based flags.

    - commissioning: before commercial operation (2016-09-01)
    - not_running: power <= min_power or rotor speed <= min_rotor_speed
    - derate_setpoint: setpoint below derate_setpoint_fraction x rated while wind is above derate_wind_speed
      (the setpoint follows the power curve at low wind, so it only indicates a derate near rated wind)
    - missing: any `required` signal is NaN
    - out_of_range: any signal outside PLAUSIBLE
    """
    f = pd.DataFrame(index=scada.index)
    f["commissioning"] = scada["time_stamp"] < COMMERCIAL_OPERATION
    f["not_running"] = ~((scada["power"] > min_power) & (scada["rotor_speed"] > min_rotor_speed))
    if "turbine_power_setpoint" in scada:
        f["derate_setpoint"] = ((scada["turbine_power_setpoint"] < derate_setpoint_fraction * RATED_POWER)
                                & (scada["wind_speed"] > derate_wind_speed)).fillna(False)
    f["missing"] = scada[list(required)].isna().any(axis=1)
    oor = np.zeros(len(scada), dtype=bool)
    for col, (lo, hi) in PLAUSIBLE.items():
        if col in scada:
            v = scada[col]
            oor |= (v.notna() & ((v < lo) | (v > hi))).to_numpy()
    # Temperature sensors report exactly 0.0 when they drop out.
    for col in scada.columns:
        if "temp" in col:
            oor |= (scada[col] == 0).to_numpy()
    f["out_of_range"] = oor
    return f


def healthy_mask(flags: pd.DataFrame, exclude) -> pd.Series:
    """True where none of the `exclude` flags is set."""
    return ~flags[list(exclude)].any(axis=1)


def period_of(times: pd.Series, split: dict) -> pd.Series:
    """Label each timestamp with the split it falls in ({name: (start, end)}, end exclusive), else 'none'."""
    out = pd.Series("none", index=times.index, dtype=object)
    for name, (start, end) in split.items():
        out[(times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))] = name
    return pd.Categorical(out, categories=[*split, "none"])
