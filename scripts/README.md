# Scripts

This directory will contain the verified scripts used to construct and audit the benchmark.

The manuscript reports three classes of executable analyses:

1. current common-cohort gap construction and matched-control generation;
2. ten-seed strict-balance sensitivity at `G=2`;
3. ten-seed physical-location leave-one-location-out (LOLO) evaluation at `G=2`.

Only scripts checked against the frozen experiment outputs should be placed here. Historical or exploratory scripts with different preprocessing, split semantics, or model inputs should not be substituted for the final audit pipeline.

The current development experiments intentionally omit the test split.
