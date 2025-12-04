
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class RAFDB_AU_Dataset(Dataset):
    """
    RAF-DB dataset with Action Units for AU-MAE pretraining.

    CSV format:
        AU01, AU02, ... AU43, filename, emotion

    Returns:
        image: Tensor [3,H,W]
        aus:   Tensor [20]
        label: int (0..6)     # emotion label
    """

    def __init__(self, csv_path, root_dir, transform=None, split="train"):
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split  # "train" or "test"

        # AU columns (all columns starting with 'AU')
        self.au_columns = [c for c in self.df.columns if c.startswith("AU")]

        print(f"[INFO] AU columns detected: {self.au_columns}")
        print(f"[INFO] Loaded {len(self.df)} AU-RAF samples (split={self.split})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        label = int(row["emotion"]) - 1

        # Use split ("train" or "test") to choose folder
        img_path = os.path.join(
            self.root_dir,
            "DATASET",
            self.split,           # "train" or "test"
            str(row["emotion"]),
            filename,
        )

        img = Image.open(img_path).convert("RGB")

        aus = torch.tensor(row[self.au_columns].values.astype("float32"))

        if self.transform:
            img = self.transform(img)

        return img, aus, label
