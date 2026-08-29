# Scripts

- `run_c3_strict_balance_10seed.py` — final strict matched-balance ten-seed rerun at G=2 s.
- `run_c4_lolo_10seed.py` — final five-location physical-location LOLO ten-seed rerun at G=2 s.
- `recompute_v4_location_statistics.py` — v4 inferential post-processing: exact binomial p-values using strict wins / all N pairs, unweighted macro PW, Cochran Q, I², and leave-one-location-out pooled stability.
- `verify_reported_metrics.py` — repository/paper consistency check for the final v4 manuscript anchors.

The two training/evaluation scripts preserve the exact final Colab path layout as provenance and contain structural assertions for the reported cohorts. The v4 statistical post-processing is intentionally separated from those archived runs because the model predictions did not change; only the inferential treatment of ties and the cross-location stability analysis changed.

To rerun the training scripts elsewhere, replace the path constants with local paths to the corresponding VIRAT-derived artifacts. Raw VIRAT video is not redistributed.
