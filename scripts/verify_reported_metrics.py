#!/usr/bin/env python3
"""Verify public repository artifacts against the final manuscript anchors."""
from pathlib import Path
import json
import pandas as pd

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

# LOLO split semantics used by the final C4 rerun.
assert lolo_cfg["gap_seconds"] == 2
assert lolo_cfg["holdout_locations"] == ["0000", "0002", "0400", "0401", "0500"]
assert lolo_cfg["pair_policy"]["frozen_before_split"] is True
assert lolo_cfg["pair_policy"]["controls_remined_within_fold"] is False
assert lolo_cfg["pair_policy"]["cross_boundary_pairs"] == "omitted"
assert lolo_cfg["inner_split"] == {
    "train_fraction": 0.85,
    "validation_fraction": 0.15,
    "seed": 20260820,
}
assert lolo_cfg["evaluation_pair_counts"] == {
    "0000": 203,
    "0002": 384,
    "0400": 277,
    "0401": 282,
    "0500": 54,
    "pooled": 1200,
}
assert lolo_cfg["test_split_loaded"] is False

# Frozen public manifests reconstructed from the final run outputs.
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

# Recompute the formal PW values in manuscript Table 3 from the final
# ten-seed averaged pair logits. This uses archived outputs, not retraining.
scores = pd.read_csv(ROOT / "results" / "current_gap_seed_averaged_pair_logits.csv.gz")
assert len(scores) == 1205
expected_gap_pw = {
    0: 0.672199,
    1: 0.680498,
    2: 0.630705,
    3: 0.522822,
    5: 0.535270,
}
for gap, expected in expected_gap_pw.items():
    part = scores[scores["gap"] == gap]
    assert len(part) == 241
    pw = float((part["pos_logit"] > part["safe_ctrl_logit"]).mean())
    assert abs(pw - expected) < 5e-7, (gap, pw, expected)

g2 = scores[scores["gap"] == 2]
assert g2["unsafe_ctrl_logit"].notna().sum() == 241
unsafe_pw = float((g2["pos_logit"] > g2["unsafe_ctrl_logit"]).mean())
assert abs(unsafe_pw - 0.639004) < 5e-7

# Headline numerical results stated in the final paper.
assert abs(reported["historical_preprocessing"]["pw_before"] - 0.9449) < 5e-7
assert abs(reported["historical_preprocessing"]["pw_after_endpoint_correction"] - 0.4959) < 5e-7
assert reported["current_gap_formal_pw"]["G2"]["n"] == 241
assert abs(reported["current_gap_formal_pw"]["G2"]["pw"] - 0.630705) < 5e-7
assert reported["lolo_G2_10seed"]["pooled"]["n"] == 1200
assert abs(reported["lolo_G2_10seed"]["pooled"]["pw"] - 0.535000) < 5e-7
assert reported["strict_balance_G2_10seed"]["validation_pairs"] == 236
assert abs(reported["strict_balance_G2_10seed"]["formal_pw"] - 0.631356) < 5e-7
assert abs(reported["future_safety_G2"]["safe_pw"] - 0.630705) < 5e-7
assert abs(reported["future_safety_G2"]["unsafe_pw"] - 0.639004) < 5e-7
assert reported["chance_calibration_n241"]["trials"] == 10000
assert abs(reported["chance_calibration_n241"]["pw_mean"] - 0.500037) < 5e-7

print("PASS: repository artifacts match the final manuscript anchors.")
print("  common cohort: 1,138 train + 241 validation")
print("  Table 3 formal PW: G0=.672199 G1=.680498 G2=.630705 G3=.522822 G5=.535270")
print("  strict-balance G2 formal PW: 0.631356 on N=236")
print("  future-safety unsafe PW: 0.639004")
print("  LOLO pooled G2 formal PW: 0.535000 on N=1200")
