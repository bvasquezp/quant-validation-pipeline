# EUR/USD source evaluation

## Decision

Use `data/raw/EURUSD_15min.csv` as the only canonical training feed. Do not
concatenate the downloaded feeds: their broker/session timestamps and volume
semantics differ. Keep the others for overlap and robustness checks.

## Canonical feed

- Source: [Hugo Lemonnier M15 EUR/USD dataset](https://www.kaggle.com/datasets/hugolemonnier/eur-15min)
- Normalized file: `data/raw/EURUSD_15min.csv`
- Coverage used: 2018-01-01 23:00 UTC to 2024-02-09 22:45 UTC
- Rows: 152,381
- Fields: OHLC plus tick volume
- Calendar: `fx_24_5`, frequency: `15min`
- SHA-256: recorded in `config/data_manifest.json`

The source has 317 normal Friday-to-Sunday market closures and 14 additional
source gaps. The latter are explicitly listed in `data.allowed_gap_dates` and
are reported in the manifest. No bar is interpolated or forward-filled.

## Overlap audit

The comparison searches 15-minute timestamp shifts in +/-12 hours and then
compares log-close returns:

| Feed | Rows | Best shift | Exact overlap | Return correlation | Decision |
|---|---:|---:|---:|---:|---|
| Godfrenam 2023-2025 | 49,710 | -60 min | 27,591 | 0.995526 | Audit only |
| Omidroshani 2000-2019 | 495,532 | -60 min | 49,585 | 0.997928 | Audit only |
| Oriola 2020-2025 | 146,902 | +120 min | 99,950 | 0.562667 | Reject: no volume |

The high correlations are useful evidence that the canonical price path is
plausible, but the required time shifts prove that silent concatenation would
be unsafe. Reproduce the audit with:

```text
python scratch/compare_eurusd_sources.py
```

## Provenance caveat

The Kaggle page for the canonical feed does not expose a clear license in the
metadata inspected. Keep it for local research until permission for public or
commercial use is confirmed. Dukascopy documents a direct EUR/USD M15 download
with UTC timestamps at [Dukascopy Node](https://www.dukascopy-node.app/instrument/eurusd);
that route was not selected as the local canonical file because the attempted
bulk download was rate-limited (HTTP 429).
