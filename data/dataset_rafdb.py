# data/dataset_rafdb.py
from .dataset_base import CSVDataset
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from models.pose_normalizer import normalize_face_pose
def build_rafdb(train_csv, val_csv, train_tfms, val_tfms, root_dir=None):
    train_ds = CSVDataset(train_csv, transform=train_tfms, root_dir=root_dir)
    val_ds   = CSVDataset(val_csv,   transform=val_tfms,   root_dir=root_dir)
    return train_ds, val_ds

class RAFDBDatasetWithAUs(Dataset):
    def __init__(self, csv_path, transform=None, au_cols=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Identify AU columns automatically if not given
        if au_cols is None:
            au_cols = [c for c in self.df.columns if c.startswith("AU")]
        self.au_cols = au_cols

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filename"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

            # Normalize face pose
        image = normalize_face_pose(image)
        # Extract AU vector
        aus = torch.tensor(row[self.au_cols].values, dtype=torch.float32)

        # Normalize AUs to [0,1] range
        aus = torch.clamp(aus, 0, 5) / 5.0  # typical OpenFace range 0–5

        return image, aus, label

    def __len__(self):
        return len(self.df)