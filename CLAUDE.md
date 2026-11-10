# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

windAnalysis is a test bed for developing and comparing fault detection algorithms on wind turbine data (e.g. SCADA / sensor time series). The repository is at an early stage; update this file as structure and conventions are established.

## Stack

- **Environment:** venv
- **Language:** Python
- **Modelling:** PyTorch
- **Visualisation:** seaborn and matplotlib

## Environment commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add new dependencies to `requirements.txt`. The only Python installed is 3.13 on an Intel Mac, and pip has no PyTorch wheel for that combination (the last Intel-macOS release is torch 2.2.2, which needs Python <=3.12 and numpy<2), so torch is commented out until a Python 3.12 venv is set up.

## Data

Dataset: "CARE to Compare" (Fraunhofer IEE) via Kaggle `azizkasimov/wind-turbine-scada-data-for-early-fault-detection`. Three wind farms (A, B, C; 86 / 257 / 957 features), 95 datasets, 10-minute SCADA data.

```bash
python data_process.py --farms A        # download + convert to data/<farm>/...parquet (Farm C is ~17 GB CSV)
```

- `data_process.py` downloads files individually via `kagglehub` (cached in `~/.cache/kagglehub`) and writes `data/<farm>/event_info.parquet`, `feature_description.parquet` and `datasets/<event_id>.parquet`.
- The Kaggle upload holds two **different** releases: semicolon-separated originals (newer; event_info has an `asset` column) and `comma_*` copies (older, different timestamps/values/status codes). Only the semicolon files are used; never mix the two.
- `loaders.py` is the read API (`load_event`, `load_event_info`, `split_train_test`, `sensor_columns`, ...).
- Each dataset is one turbine: a `train` year followed by a `prediction` period containing one event. For anomaly events, rows with `id` between `event_start_id` and `event_end_id` are the fault window (`is_anomaly`). `status_type_id` 0 and 2 are normal operation (`is_normal_status`).

## Design intent

As a test bed, the code should make it easy to run several fault detection algorithms against the same data and compare them on equal terms. Favour keeping these concerns separate:

- data loading / preprocessing of turbine data
- the detection algorithms themselves (classical baselines and PyTorch models)
- evaluation (shared metrics, so results across algorithms are comparable)
- plotting of signals, detections, and results

`data/` and `.venv/` are git-ignored.
