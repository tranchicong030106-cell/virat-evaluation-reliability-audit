# Manifests

This directory documents the frozen identifiers used by the benchmark audit.

## Common cohort

The current common cohort is formed by intersecting actor-event keys that are eligible at every evaluated anticipation gap `G in {0,1,2,3,5}` seconds. The development package contains 1,138 training pairs and 241 validation pairs. The test split is intentionally not loaded during development experiments.

## Physical location

Physical location is parsed from the first four digits after `VIRAT_S_` in the clip identifier. The common cohort contains the five locations:

`0000`, `0002`, `0400`, `0401`, `0500`.

Six-digit identifiers correspond to camera views and are not treated as physical-location IDs.

## Frozen matched pairs

The LOLO analysis uses frozen positive-control pairs. A pair is evaluated in a held-out fold only when both positive and control belong to the held-out physical location. Cross-boundary pairs are omitted. Controls are not re-mined independently inside each fold.

The full frozen pair manifests are being prepared for the archival release. They are not regenerated from the summary numbers in this repository.
