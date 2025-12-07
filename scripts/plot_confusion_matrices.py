import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
LOG_DIR = "logs/checkpoints_baseline"
OUT_DIR = "logs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# RAF-DB class names (adjust if needed)
classes = [
    "Surprise", "Fear", "Disgust", "Happiness",
    "Sadness", "Anger", "Neutral"
]

def plot_cm(cm, title, out_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# --- Ablation A ---
cm_A = np.load(os.path.join(LOG_DIR, "ablation_A_no_pose_confusion.npy"))
plot_cm(
    cm_A,
    "Ablation A: FER Baseline",
    os.path.join(OUT_DIR, "ablation_A_confusion_matrix.png")
)

# --- Ablation B ---
cm_B = np.load(os.path.join(LOG_DIR, "ablation_B_pose_confusion.npy"))
plot_cm(
    cm_B,
    "Ablation B: FER + Pose Normalization",
    os.path.join(OUT_DIR, "ablation_B_confusion_matrix.png")
)

# --- Ablation C ---
cm_C = np.load(os.path.join(LOG_DIR, "ablation_C_au_mae_confusion.npy"))
plot_cm(
    cm_C,
    "Ablation C: AU-MAE Fine-Tuning",
    os.path.join(OUT_DIR, "ablation_C_confusion_matrix.png")
)

print("✅ Confusion matrix images saved to:", OUT_DIR)
