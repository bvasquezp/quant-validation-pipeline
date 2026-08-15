# EUR/USD data sources

`EURUSD_15min.csv` is the canonical training file. It is normalized from the
public M15 dataset by `scratch/normalize_eurusd.py` and covers the configured
development/OOS windows through 2024-02-09.

The pipeline uses one canonical feed. Other downloaded datasets are retained
only for overlap checks; their broker/session conventions and volumes must not
be silently concatenated with the canonical feed.

Provenance is recorded in `config/eurusd_source.json` and the pipeline writes a
content hash to `config/data_manifest.json` when the real CSV is loaded.

The selected feed has 317 normal Friday-to-Sunday FX closures and 14 explicit
source-gap windows in the configured range. Those windows are allowlisted in
the config only to preserve the raw observations; no prices or volumes are
filled or interpolated. A different feed requires a new quality review.
