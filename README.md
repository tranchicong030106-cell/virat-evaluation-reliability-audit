# VIRAT Evaluation Reliability Audit

Reproducibility artifacts for the paper **Auditing Evaluation Reliability in Surveillance Video Anticipation on VIRAT**.

This repository accompanies a protocol-audit study of person-centric activity anticipation on the VIRAT surveillance dataset. The study focuses on evaluation reliability rather than proposing a new anticipation architecture; Motion TCN is used as a predictive probe.

## Dataset provenance

Video data come from the **VIRAT Ground Camera Dataset, Release 2.0**. The benchmark uses the public extended VIRAT annotations released by Kitware under the IARPA DIVA program. The local annotation mirror used for the study contains the public `train` and `validate` partitions in KPF-style YAML. Raw VIRAT videos are **not redistributed** and remain subject to their original access terms.

Official resources:

- VIRAT data portal: https://viratdata.org/
- Extended VIRAT annotations: https://gitlab.kitware.com/viratdata/viratannotations

## Current benchmark

- observation duration: 3 s (90 frames at 30 fps)
- anticipation gaps: `{0, 1, 2, 3, 5}` s
- target activities: `vehicle_starting`, `vehicle_stopping`, `vehicle_turning_left`, `vehicle_turning_right`, `Opening`, `Closing`, `Entering`, `Interacts`
- common development cohort: 1,138 train pairs and 241 validation pairs
- current model: Motion TCN used as a predictive probe, not as a proposed architecture
- canonical seeds: `42, 1337, 2024, 7, 17, 73, 101, 314, 777, 1729`

Physical location is defined by the first four digits after `VIRAT_S_` in a clip identifier. The common cohort contains five physical locations: `0000`, `0002`, `0400`, `0401`, and `0500`. Six-digit identifiers are treated as camera views, not physical locations.

## Predictive probe configuration

The final Motion TCN uses six motion channels, three residual causal temporal-convolution blocks with dilations `1, 2, 4`, hidden width `80`, dropout `0.20`, temporal mean pooling, and a linear head (`98,561` parameters). Training uses AdamW (`lr=1e-3`, weight decay `1e-4`), batch size `64`, gradient clipping at `1`, early stopping on validation AP with patience `8`, and at most `50` epochs. Feature normalization is estimated from training data only.

## v4 inference convention

PW is defined as the number of strict pairwise wins divided by all evaluated pairs. The v4 two-sided exact binomial tests use the same denominator `N`; ties are therefore treated as non-wins rather than being dropped from the test denominator. This keeps the reported p-values consistent with the reported PW and `N`.

## Final headline results

| Audit | Result |
|---|---|
| Historical preprocessing correction | PW `0.9449 -> 0.4959` on `N=121` |
| Current development, G=2 s | formal PW `0.6307`, `N=241` |
| Physical-location LOLO, G=2 s | descriptive pooled PW `0.5350`, 95% CI `[0.5067, 0.5633]`, exact p `0.0165`, `N=1200` |
| Location heterogeneity | Cochran Q `25.77` (`df=4`), I² `84.5%`, unweighted macro PW `0.5688` |
| Strict matched-balance sensitivity | mean PW `0.6141 -> 0.6140`; strict formal PW `0.6314`, `N=236` |
| Future-safety sensitivity | safe `0.6307` vs unsafe `0.6390`; paired CI for delta spans zero |
| Chance calibration | PW mean `0.5000`; empirical 95% interval `[0.4357, 0.5643]` on `N=241` |

The physical-location result is highly heterogeneous rather than a stable transferable effect: location `0400` is significantly below chance, while `0401` and `0500` are significantly above. A leave-one-location-out stability check shows that removing any single location except `0400` makes the pooled 95% interval cover `0.5`. The pooled `0.5350` is therefore reported as a descriptive summary, not as evidence of a consistent location-transferable effect.

The 3 s and 5 s development results are not statistically distinguishable from chance on the current `N=241` validation cohort.

## Frozen reproducibility artifacts

`manifests/` documents and stores the frozen pair identities and LOLO evaluation split labels used by the manuscript. `results/` stores machine-readable reported values and compact audit summaries. Frozen controls are not re-mined inside LOLO folds.

The exact final scripts used for the two advisor-requested ten-seed reruns are included:

- `scripts/run_c4_lolo_10seed.py` — frozen-pair physical-location LOLO at G=2 s;
- `scripts/run_c3_strict_balance_10seed.py` — strict matched-balance sensitivity at G=2 s.

The v4 inferential post-processing is separated explicitly from those archived training runs:

- `scripts/recompute_v4_location_statistics.py` — recomputes exact binomial p-values, unweighted macro PW, Cochran Q, I², and leave-one-location-out stability from the frozen LOLO summary;
- `results/lolo_location_summary_v4.csv` — location-level counts, strict wins, ties, PW, CI, and corrected exact p-values;
- `results/lolo_leave_one_location_out_v4.csv` — leave-one-location-out pooled sensitivity;
- `results/lolo_location_stability_v4.json` — machine-readable v4 stability summary.

## Verification

Install the lightweight environment and run:

```bash
pip install -r requirements.txt
python scripts/recompute_v4_location_statistics.py
python scripts/verify_reported_metrics.py
```

The verifier checks repository configuration and the v4 manuscript anchors without retraining the model.

## Repository layout

```text
configs/      frozen experiment constants and protocol settings
manifests/    frozen pair identities and LOLO evaluation split labels
scripts/      verified audit scripts and v4 statistical post-processing
results/      machine-readable reported results and stability summaries
docs/         source and bibliography audit notes
refs.bib      source-audited manuscript bibliography
```

## Reference audit

`docs/reference_audit.md` records the source verification used for the bibliography, including the IEEE/DBLP versus CVF pagination differences for several conference papers. The repository bibliography follows one consistent DOI/IEEE/DBLP convention.

## Citation

A citation entry will be added after publication metadata are available.
