#!/usr/bin/env python3
"""Recompute v4 location-level inference from the ten-seed LOLO summary.

The manuscript defines PW as strict pairwise wins divided by all evaluated pairs.
For v4, the two-sided exact binomial test uses that same denominator N; ties are
therefore non-wins rather than being dropped from the test denominator.
"""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "lolo_location_summary_v4.csv"
OUT_LOO = ROOT / "results" / "lolo_leave_one_location_out_v4.csv"
OUT_JSON = ROOT / "results" / "lolo_location_stability_v4.json"

loc = pd.read_csv(SUMMARY, dtype={"location": str})
base = loc[loc["location"] != "POOLED"].copy()
base["location"] = base["location"].str.zfill(4)
expected = ["0000", "0002", "0400", "0401", "0500"]
assert base["location"].tolist() == expected
assert base["N"].tolist() == [203, 384, 277, 282, 54]
assert base["wins"].tolist() == [113, 207, 120, 162, 40]

# Recompute PW and exact two-sided binomial p-values from strict wins / N.
base["PW_recomputed"] = base["wins"] / base["N"]
base["p_recomputed"] = [
    binomtest(int(w), int(n), p=0.5, alternative="two-sided").pvalue
    for w, n in zip(base["wins"], base["N"])
]
assert np.allclose(base["PW_recomputed"], base["PW"], atol=5e-7)
assert np.allclose(base["p_recomputed"], base["p_two_sided_exact_binomial"], atol=5e-12)

# Unweighted macro PW across physical locations.
macro_pw = float(base["PW"].mean())

# Cochran Q / I^2 using inverse-variance weights for location-level proportions.
p = base["PW"].to_numpy(float)
n = base["N"].to_numpy(float)
var = p * (1.0 - p) / n
weights = 1.0 / var
p_fe = float(np.sum(weights * p) / np.sum(weights))
Q = float(np.sum(weights * (p - p_fe) ** 2))
df = len(base) - 1
I2 = float(max(0.0, (Q - df) / Q) * 100.0)
Q_p = float(chi2.sf(Q, df))

# Descriptive pooled strict-win PW.
total_n = int(base["N"].sum())
total_wins = int(base["wins"].sum())
pooled_pw = total_wins / total_n
pooled_p = float(binomtest(total_wins, total_n, p=0.5, alternative="two-sided").pvalue)
assert total_n == 1200 and total_wins == 642
assert abs(pooled_pw - 0.5350) < 5e-12

# Leave-one-location-out pooled summaries. CIs follow the normal approximation
# requested for the v4 sensitivity check.
loo_rows = []
for _, row in base.iterrows():
    N = total_n - int(row["N"])
    W = total_wins - int(row["wins"])
    pw = W / N
    se = math.sqrt(pw * (1.0 - pw) / N)
    loo_rows.append({
        "omitted_location": row["location"],
        "N": N,
        "wins": W,
        "PW": pw,
        "normal_CI95_low": pw - 1.96 * se,
        "normal_CI95_high": pw + 1.96 * se,
        "p_two_sided_exact_binomial": float(
            binomtest(W, N, p=0.5, alternative="two-sided").pvalue
        ),
    })
loo = pd.DataFrame(loo_rows)
loo.to_csv(OUT_LOO, index=False)

result = {
    "macro_unweighted_pw": macro_pw,
    "pooled": {
        "N": total_n,
        "wins": total_wins,
        "PW": pooled_pw,
        "p_two_sided_exact_binomial": pooled_p,
    },
    "heterogeneity": {
        "cochran_Q": Q,
        "df": df,
        "Q_p_value": Q_p,
        "I2_percent": I2,
        "fixed_effect_mean_used_for_Q": p_fe,
    },
    "leave_one_location_out": loo_rows,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Macro PW: {macro_pw:.4f}")
print(f"Cochran Q={Q:.2f}, df={df}, I^2={I2:.1f}%")
print(f"Pooled PW={pooled_pw:.4f}, exact p={pooled_p:.4f}")
print(loo.to_string(index=False))
