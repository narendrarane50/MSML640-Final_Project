import json
import os
import numpy as np

def save_metrics(metrics, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)

    # Save scalar metrics
    clean = {k: float(v) for k, v in metrics.items() if k != "confusion_matrix"}
    with open(os.path.join(save_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(clean, f, indent=2)

    # Save confusion matrix
    if "confusion_matrix" in metrics:
        np.save(
            os.path.join(save_dir, f"{tag}_confusion.npy"),
            metrics["confusion_matrix"]
        )
