import os, csv

base_dir = "datasets/RAF-DB/DATASET"
output_dir = "splits"
os.makedirs(output_dir, exist_ok=True)

label_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}

def create_split(split_name):
    img_root = os.path.join(base_dir, split_name)
    out_csv = os.path.join(output_dir, f"rafdb_{'val' if split_name == 'test' else 'train'}.csv")
    rows = []

    for lbl_folder, lbl_id in label_map.items():
        folder = os.path.join(img_root, lbl_folder)
        if not os.path.exists(folder):
            print(f"[!] Skipping missing folder: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                
                rel_path = os.path.join("DATASET", split_name, lbl_folder, fname)
                rows.append([rel_path, lbl_id])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"[✓] Created {out_csv} with {len(rows)} images.")

create_split("train")
create_split("test")
