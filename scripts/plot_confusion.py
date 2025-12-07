import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--cm", required=True)
ap.add_argument("--title", default="Confusion Matrix")
ap.add_argument("--out", required=True)
args = ap.parse_args()

cm = np.load(args.cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(args.title)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(args.out)
plt.close()
