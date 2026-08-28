# Frozen manifests

This directory contains the frozen pair identities and evaluation split labels used by the final manuscript analyses.

Pair identity is represented by the unique actor-event key used by the final pipeline. The common development cohort contains 1,138 training pairs and 241 validation pairs. The strict spatial-balance sensitivity removes five validation pairs and retains 236. The G=2 physical-location LOLO evaluation contains 1,200 evaluated pairs across five four-digit physical locations.

Expected artifacts:

- `common_cohort_ids.csv`: all 1,379 common-cohort pair identities with `train`/`val` split labels.
- `strict_balance_removed_ids.csv`: the five validation pair identities removed by the strict-balance sensitivity.
- `lolo_g2_eval_ids.csv`: the 1,200 G=2 LOLO evaluation pair identities with the held-out four-digit location.

The five physical locations are `0000`, `0002`, `0400`, `0401`, and `0500`. Six-digit VIRAT identifiers are camera views and are not treated as physical locations. LOLO uses frozen matched pairs: controls are not re-mined independently inside folds, and cross-boundary pairs are omitted.

Raw VIRAT video is not redistributed. The final training scripts retain structural assertions for the expected pair cohorts, while `scripts/verify_reported_metrics.py` checks the public repository configuration and, when the pair-level artifacts are present, recomputes the principal formal PW values from archived ten-seed averaged logits.
