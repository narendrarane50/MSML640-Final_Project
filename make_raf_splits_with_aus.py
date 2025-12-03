"""
make_raf_splits_with_aus.py
------------------------------------------------------------
Merges existing RAF-DB train/val split CSVs with AU features
as specified in config_aus.yaml.

Expected inputs (defined in config_aus.yaml):
  - train_split_csv
  - val_split_csv
  - au_feature_csv

Outputs:
  - train_split_with_aus
  - val_split_with_aus
"""

import os
import pandas as pd
import yaml


# -------------------------------
# Load configuration
# -------------------------------
def load_config_aus(path="config_aus.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[!] Could not find configuration file: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------
# Merge helper function
# -------------------------------
def merge_with_aus(split_csv, au_csv, out_csv):
    print(f"[*] Merging: {split_csv} with {au_csv}")
    split_df = pd.read_csv(split_csv, names=["path", "label"])
    au_df = pd.read_csv(au_csv)

    if "path" not in au_df.columns:
        raise ValueError("AU CSV must contain a 'path' column")

    # Normalize paths (relative to RAF-DB root)
    au_df["path"] = au_df["path"].apply(lambda p: os.path.relpath(p, start="datasets/RAF-DB"))

    # Merge
    merged = split_df.merge(au_df, on="path", how="inner")
    merged.to_csv(out_csv, index=False)

    print(f"[✓] Wrote {out_csv}")
    print(f"    Samples: {len(merged)}, AU features: {len(merged.columns) - 2}\n")


# -------------------------------
# Main
# -------------------------------
def main():
    cfg = load_config_aus("config_aus.yaml")

    train_csv = cfg["train_split_csv"]
    val_csv = cfg["val_split_csv"]
    au_csv = cfg["au_feature_csv"]
    train_out = cfg["train_split_with_aus"]
    val_out = cfg["val_split_with_aus"]

    # Sanity checks
    for fpath in [train_csv, val_csv, au_csv]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[!] Missing required file: {fpath}")

    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    print("[*] Starting AU-augmented split creation...")
    merge_with_aus(train_csv, au_csv, train_out)
    merge_with_aus(val_csv, au_csv, val_out)
    print("[✔] Done — AU-augmented splits created successfully.")


if __name__ == "__main__":
    main()
