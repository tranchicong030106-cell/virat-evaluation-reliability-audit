# VIRAT Evaluation Reliability Audit

Reproducibility artifacts for the paper **Auditing Evaluation Reliability in Surveillance Video Anticipation on VIRAT**.

This repository accompanies a protocol-audit study of person-centric activity anticipation on the VIRAT surveillance dataset. The study focuses on evaluation reliability rather than proposing a new anticipation architecture.

## What is audited

The paper evaluates six aspects of benchmark reliability:

1. temporal preprocessing consistency;
2. finite-cohort chance calibration;
3. physical-location separation;
4. anticipation-gap sensitivity;
5. matched-control balance;
6. future-safety sensitivity.

## Dataset provenance

Video data come from the **VIRAT Ground Camera Dataset, Release 2.0**. The benchmark uses the public extended VIRAT annotations released by Kitware under the IARPA DIVA program. The raw VIRAT videos are **not redistributed** in this repository and remain subject to their original access terms.

Official resources:

- VIRAT data portal: https://viratdata.org/
- Extended VIRAT annotations: https://gitlab.kitware.com/viratdata/viratannotations

The local annotation mirror used in the study contains 119 annotated clips across the public `train` and `validate` partitions, with activity, geometry, and actor-type annotations stored in KPF-style YAML files.

## Current benchmark

- observation duration: 3 s (90 frames at 30 fps)
- anticipation gaps: `{0, 1, 2, 3, 5}` s
- target activities: `vehicle_starting`, `vehicle_stopping`, `vehicle_turning_left`, `vehicle_turning_right`, `Opening`, `Closing`, `Entering`, `Interacts`
- common development cohort: 1,138 train pairs and 241 validation pairs
- current model: Motion TCN used as a predictive probe, not as a proposed architecture
- canonical seeds: `42, 1337, 2024, 7, 17, 73, 101, 314, 777, 1729`

Physical location is defined by the first four digits after `VIRAT_S_` in a clip identifier. The common cohort contains five physical locations: `0000`, `0002`, `0400`, `0401`, and `0500`. Six-digit identifiers are treated as camera views, not physical locations.

## Final headline results

| Audit | Result |
|---|---|
| Historical preprocessing correction | PW `0.9449 -> 0.4959` on `N=121` |
| Current development, G=2 s | formal PW `0.6307`, `N=241` |
| Physical-location LOLO, G=2 s | pooled formal PW `0.5350`, 95% CI `[0.5067, 0.5633]`, `N=1200`, `p=0.0040` |
| Strict matched-balance sensitivity | mean PW `0.6141 -> 0.6140`; strict formal PW `0.6314`, `N=236` |
| Future-safety sensitivity | safe `0.6307` vs unsafe `0.6390`; paired CI for delta spans zero |

The physical-location result should be interpreted as a **sensitivity analysis**, not a causal estimate of scene memorization. The 3 s and 5 s development results are not statistically distinguishable from chance on the current `N=241` development cohort.

## Reproducing the two final advisor-requested reruns

The exact final scripts used to synchronize the main sensitivity results to the ten canonical seeds are now included:

- `scripts/run_c4_lolo_10seed.py` — frozen-pair physical-location LOLO at G=2 s;
- `scripts/run_c3_strict_balance_10seed.py` — strict matched-balance sensitivity at G=2 s.

Both scripts contain structural assertions for the expected cohorts and do not load the test split.

## Repository layout

```text
configs/      experiment constants and protocol settings
manifests/    common-cohort identifiers and manifest documentation
scripts/      verified audit scripts
results/      compact machine-readable summaries of reported results
docs/         source and bibliography audit notes
refs.bib      source-audited manuscript bibliography
```

## Reference audit

`docs/reference_audit.md` records the source verification used for the bibliography, including the IEEE/DBLP versus CVF pagination differences for several conference papers. The repository bibliography follows one consistent DOI/IEEE/DBLP convention.

## Reproducibility status

The two final ten-seed rerun scripts, frozen experiment configuration, result summaries, bibliography, and source-audit notes are public. Raw VIRAT videos are not included. Additional frozen pair manifests may be added to the archival release after final verification.

## Citation

A citation entry will be added after publication metadata are available.
