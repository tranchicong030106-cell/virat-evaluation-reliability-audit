#!/usr/bin/env python3
"""Verify public repository artifacts against the final v4 manuscript anchors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "experiment_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
with open(ROOT / "configs" / "lolo_g2_split_config.json", "r", encoding="utf-8") as f:
    lolo_cfg = json.load(f)
with open(ROOT / "results" / "final_results.json", "r", encoding="utf-8") as f:
    reported = json.load(f)

# Protocol/model values stated in the manuscript.
assert cfg["dataset"]["release"] == "2.0"
assert cfg["observation"] == {
    "duration_seconds": 3,
    "fps": 30,
    "frames": 90,
    "interval_convention": "half-open",
}
assert cfg["gaps_seconds"] == [0, 1, 2, 3, 5]
assert cfg["current_development_cohort"]["train_pairs"] == 1138
assert cfg["current_development_cohort"]["validation_pairs"] == 241
assert cfg["current_development_cohort"]["test_loaded_for_development"] is False
assert cfg["physical_locations"] == ["0000", "0002", "0400", "0401", "0500"]
assert cfg["model"]["name"] == "Motion TCN"
assert cfg["model"]["input_channels"] == 6
assert cfg["model"]["hidden_width"] == 80
assert cfg["model"]["dilations"] == [1, 2, 4]
assert cfg["model"]["kernel_size"] == 3
assert cfg["model"]["dropout"] == 0.2
assert cfg["model"]["parameters"] == 98561
assert cfg["training"]["optimizer"] == "AdamW"
assert cfg["training"]["learning_rate"] == 0.001
assert cfg["training"]["weight_decay"] == 0.0001
assert cfg["training"]["batch_size"] == 64
assert cfg["training"]["max_epochs"] == 50
assert cfg["training"]["gradient_clip"] == 1.0
assert cfg["training"]["early_stopping_metric"] == "validation AP"
assert cfg["training"]["early_stopping_patience"] == 8
assert cfg["training"]["feature_normalization"] == "training data only"
assert cfg["seeds"] == [42, 1337, 2024, 7, 17, 73, 101, 314, 777, 1729]

# LOLO split semantics used by the final ten-seed rerun.
assert lolo_cfg["gap_seconds"] == 2
assert lolo_cfg["holdout_locations"] == ["0000", "0002", "0400", "0401", "0500"]
assert lolo_cfg["pair_policy"]["frozen_before_split"] is True
assert lolo_cfg["pair_policy"]["controls_remined_within_fold"] is False
assert lolo_cfg["pair_policy"]["cross_boundary_pairs"] == "omitted"
assert lolo_cfg["evaluation_pair_counts"] == {
    "0000": 203,
    "0002": 384,
    "0400": 277,
    "0401": 282,
    "0500": 54,
    "pooled": 1200,
}
assert lolo_cfg["test_split_loaded"] is False

# Frozen public manifests.
common = pd.read_csv(ROOT / "manifests" / "common_cohort_ids.csv.gz")
assert len(common) == 1379
assert common["event_key"].nunique() == 1379
assert int((common["split"] == "train").sum()) == 1138
assert int((common["split"] == "val").sum()) == 241

removed = pd.read_csv(ROOT / "manifests" / "strict_balance_removed_ids.csv")
assert len(removed) == 5
assert (removed["split"] == "val").all()
assert removed["target_class"].value_counts().to_dict() == {
    "vehicle_stopping": 2,
    "Closing": 2,
    "vehicle_starting": 1,
}

lolo_ids = pd.read_csv(
    ROOT / "manifests" / "lolo_g2_eval_ids.csv.gz",
    dtype={"holdout_location": str},
)
lolo_ids["holdout_location"] = lolo_ids["holdout_location"].str.zfill(4)
assert len(lolo_ids) == 1200
assert lolo_ids.groupby("holdout_location").size().to_dict() == {
    "0000": 203,
    "0002": 384,
    "0400": 277,
    "0401": 282,
    "0500": 54,
}

# Recompute Table 3 PW and v4 exact-binomial p-values from archived averaged logits.
scores = pd.read_csv(ROOT / "results" / "current_gap_seed_averaged_pair_logits.csv.gz")
assert len(scores) == 1205
expected_gap = {
    0: (162, 0.672199),
    1: (164, 0.680498),
    2: (152, 0.630705),
    3: (126, 0.522822),
    5: (129, 0.535270),
}
for gap, (expected_wins, expected_pw) in expected_gap.items():
    part = scores[scores["gap"] == gap]
    assert len(part) == 241
    wins = int((part["pos_logit"] > part["safe_ctrl_logit"]).sum())
    pw = wins / len(part)
    p = float(binomtest(wins, len(part), p=0.5, alternative="two-sided").pvalue)
    assert wins == expected_wins
    assert abs(pw - expected_pw) < 5e-7
    assert abs(p - reported["current_gap_formal_pw"][f"G{gap}"]["p_two_sided"]) < 5e-12

# Future-safety headline value.
g2 = scores[scores["gap"] == 2]
assert g2["unsafe_ctrl_logit"].notna().sum() == 241
unsafe_pw = float((g2["pos_logit"] > g2["unsafe_ctrl_logit"]).mean())
assert abs(unsafe_pw - 0.639004) < 5e-7

# v4 location-level summary and corrected p-values.
loc = pd.read_csv(ROOT / "results" / "lolo_location_summary_v4.csv", dtype={"location": str})
base = loc[loc["location"] != "POOLED"].copy()
base["location"] = base["location"].str.zfill(4)
assert base["location"].tolist() == ["0000", "0002", "0400", "0401", "0500"]
assert base["N"].tolist() == [203, 384, 277, 282, 54]
assert base["wins"].tolist() == [113, 207, 120, 162, 40]
for _, row in base.iterrows():
    p = float(binomtest(int(row["wins"]), int(row["N"]), p=0.5, alternative="two-sided").pvalue)
    assert abs(p - float(row["p_two_sided_exact_binomial"])) < 5e-12

pooled = loc[loc["location"] == "POOLED"].iloc[0]
assert int(pooled["N"]) == 1200 and int(pooled["wins"]) == 642
assert abs(float(pooled["PW"]) - 0.5350) < 5e-12
assert abs(float(pooled["p_two_sided_exact_binomial"]) - 0.0165392328534717) < 5e-12

stability = reported["lolo_G2_10seed"]
assert abs(stability["macro_unweighted_pw"] - 0.5688269137084868) < 5e-12
assert abs(stability["heterogeneity"]["cochran_Q"] - 25.76610541377163) < 5e-10
assert stability["heterogeneity"]["df"] == 4
assert abs(stability["heterogeneity"]["I2_percent"] - 84.47572911868141) < 5e-10

loo = pd.read_csv(ROOT / "results" / "lolo_leave_one_location_out_v4.csv", dtype={"omitted_location": str})
loo["omitted_location"] = loo["omitted_location"].str.zfill(4)
expected_loo = {
    "0000": (997, 0.5305917753, 0.4996130106, 0.5615705401),
    "0002": (816, 0.5330882353, 0.4988565761, 0.5673198945),
    "0400": (923, 0.5655471289, 0.5335684177, 0.5975258402),
    "0401": (918, 0.5228758170, 0.4905648673, 0.5551867667),
    "0500": (1146, 0.5253054101, 0.4963934963, 0.5542173239),
}
for _, row in loo.iterrows():
    n, pw, lo, hi = expected_loo[row["omitted_location"]]
    assert int(row["N"]) == n
    assert abs(float(row["PW"]) - pw) < 5e-10
    assert abs(float(row["normal_CI95_low"]) - lo) < 5e-10
    assert abs(float(row["normal_CI95_high"]) - hi) < 5e-10

# Remaining headline anchors.
assert abs(reported["historical_preprocessing"]["pw_before"] - 0.9449) < 5e-7
assert abs(reported["historical_preprocessing"]["pw_after_endpoint_correction"] - 0.4959) < 5e-7
assert reported["strict_balance_G2_10seed"]["validation_pairs"] == 236
assert abs(reported["strict_balance_G2_10seed"]["formal_pw"] - 0.631356) < 5e-7
assert abs(reported["future_safety_G2"]["safe_pw"] - 0.630705) < 5e-7
assert abs(reported["future_safety_G2"]["unsafe_pw"] - 0.639004) < 5e-7
assert reported["chance_calibration_n241"]["trials"] == 10000
assert abs(reported["chance_calibration_n241"]["pw_mean"] - 0.500037) < 5e-7

print("PASS: repository artifacts match the final v4 manuscript anchors.")
print("  common cohort: 1,138 train + 241 validation")
print("  Table 3 exact-binomial p: G3=.5196, G5=.3027")
print("  LOLO pooled: PW=.5350, exact p=.0165")
print("  LOLO heterogeneity: Q=25.77, df=4, I^2=84.5%, macro=.5688")
print("  leave-one-location-out: only omission of 0400 keeps the 95% CI above .5")
