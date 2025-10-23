import os
import pandas as pd
from feat import Detector
from tqdm import tqdm

# Paths
dataset_root = "datasets/RAF-DB/DATASET"
output_csv = "datasets/RAF-DB/aus_features.csv"

# Initialize FEAT detector (this downloads pretrained models automatically)
detector = Detector(
    face_model="retinaface",
    landmark_model="mobilenet",
    au_model="rf",        # or "svm"
    emotion_model="resmasknet"
)

rows = []

# Loop over all subfolders (1–7 emotions)
for split in ["train", "test"]:
    for root, _, files in os.walk(os.path.join(dataset_root, split)):
        for f in tqdm(files, desc=f"Processing {split}"):
            if not f.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(root, f)
            rel_path = os.path.relpath(img_path, start="datasets/RAF-DB")
            try:
                result = detector.detect_image(img_path)
                aus = result.aus.iloc[0].to_dict()
                row = {"path": rel_path, **aus}
                rows.append(row)
            except Exception as e:
                print(f"[!] Failed {img_path}: {e}")

# Save
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"[✓] Saved AU features to {output_csv} ({len(df)} rows, {len(df.columns)-1} AU columns)")
