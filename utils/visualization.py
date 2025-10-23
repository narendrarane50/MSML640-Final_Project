"""
visualization.py
------------------------------------------------------------
Generic visualization utility for training metrics.
Usage:
    python visualization.py --csv checkpoints_baseline/metrics_log.csv --out_dir results/
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(csv_path, out_dir="results", show=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["accuracy"], label="Accuracy", marker="o")
    plt.plot(df["epoch"], df["balanced_accuracy"], label="Balanced Accuracy", marker="s")
    plt.plot(df["epoch"], df["macro_f1"], label="Macro F1", marker="^")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_metrics.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"[✔] Visualizations saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to metrics_log.csv")
    parser.add_argument("--out_dir", default="results", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    visualize_metrics(args.csv, args.out_dir, args.show)
