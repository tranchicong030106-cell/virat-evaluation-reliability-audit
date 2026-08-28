# Scripts

- `run_c3_strict_balance_10seed.py` — final strict matched-balance sensitivity at G=2 s.
- `run_c4_lolo_10seed.py` — final five-location physical-location LOLO sensitivity at G=2 s.
- `verify_reported_metrics.py` — repository/paper consistency check; when compact pair-level averaged-logit artifacts are present, it also recomputes the principal formal PW values without retraining.

The two final training/evaluation scripts preserve the exact final Colab path layout as provenance and contain structural assertions for the cohorts reported in the manuscript. To rerun them elsewhere, replace the path constants with local paths to the corresponding VIRAT-derived artifacts. Raw VIRAT video is not redistributed.
