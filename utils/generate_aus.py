import os
import argparse
import pandas as pd
from feat import Detector
import torch
import time
import glob

def extract_aus_for_emotion(detector, emotion_path, emotion_label, split_name, out_csv, save_every=500):
    """Extract AUs for one emotion folder and save incrementally."""
    if os.path.exists(out_csv):
        print(f"[SKIP] {out_csv} already exists, skipping emotion {emotion_label}")
        return

    images = [f for f in os.listdir(emotion_path)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"[INFO] Processing emotion {emotion_label} ({len(images)} images)")

    buffer = []
    done_count = 0

    for idx, img_name in enumerate(images):
        img_path = os.path.join(emotion_path, img_name)
        try:
            result = detector.detect_image(img_path)
            aus = result.aus.iloc[0].to_dict()
            aus["filename"] = img_name
            aus["emotion"] = int(emotion_label)
            buffer.append(aus)
            done_count += 1
        except Exception as e:
            print(f"[WARN] Skipped {img_name}: {e}")
            continue

        # Save periodically
        if done_count % save_every == 0:
            pd.DataFrame(buffer).to_csv(
                out_csv,
                mode='a',
                header=not os.path.exists(out_csv),
                index=False
            )
            buffer.clear()
            print(f"  └── Saved {done_count}/{len(images)} images for emotion {emotion_label}")
            time.sleep(0.1)

    # Flush remaining
    if buffer:
        pd.DataFrame(buffer).to_csv(
            out_csv,
            mode='a',
            header=not os.path.exists(out_csv),
            index=False
        )
        print(f"[SAVE] Completed emotion {emotion_label}: {done_count} total images.")

def extract_aus_per_emotion(split_dir, split_name, out_dir, target_emotion=None, save_every=500):
    """Process all or a specific emotion folder."""
    os.makedirs(out_dir, exist_ok=True)

    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model='resmasknet',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    for emotion_label in sorted(os.listdir(split_dir)):
        if not emotion_label.isdigit():
            continue
        if target_emotion and int(emotion_label) != target_emotion:
            continue

        emotion_path = os.path.join(split_dir, emotion_label)
        if not os.path.isdir(emotion_path):
            continue

        out_csv = os.path.join(out_dir, f"{split_name}_emotion{emotion_label}.csv")
        extract_aus_for_emotion(detector, emotion_path, emotion_label, split_name, out_csv, save_every)

    print(f"[DONE] Completed split {split_name} ({'single emotion' if target_emotion else 'all emotions'})")


def merge_emotions_aus_files(in_loc:str, out_loc:str):
    folder = in_loc
    all_csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    
    if not all_csvs:
        print(f"[ERROR] No CSVs found in {folder}")
        return
    
    valid_csvs = [f for f in all_csvs if os.path.getsize(f) > 0]
    if not valid_csvs:
        print(f"[ERROR] All CSVs in {folder} are empty.")
        return
    
    df_list = [pd.read_csv(f) for f in valid_csvs]
    merged = pd.concat(df_list, ignore_index=True)
    
    out_path = out_loc
    merged.to_csv(out_path, index=False)
    print(f"[MERGED] → {out_path} ({len(merged)} entries)")


loc = '/kaggle/working/MSML640-Final_Project/data/RAFDB/splits/emotion_csvs/test'
out_loc = '/kaggle/working/MSML640-Final_Project/splits/aus/train/aus_raf_test.csv'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract AU features per emotion folder")
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True,
                        help="Dataset split to process (train or test)")
    parser.add_argument("--emotion", type=int, default=None,
                        help="If provided, process only this emotion folder (e.g., 3)")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Number of images before appending to CSV")
    args = parser.parse_args()

    base_dir = "datasets/RAF-DB/DATASET"
    split_dir = os.path.join(base_dir, args.split)
    out_dir = f"data/RAFDB/splits/emotion_csvs/{args.split}"

    extract_aus_per_emotion(split_dir, f"raf_{args.split}", out_dir,
                            target_emotion=args.emotion, save_every=args.save_every)
