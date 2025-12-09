
from .dataset_base import CSVDataset
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
def build_rafdb(train_csv, val_csv, train_tfms, val_tfms, root_dir=None):
    train_ds = CSVDataset(train_csv, transform=train_tfms, root_dir=root_dir)
    val_ds   = CSVDataset(val_csv,   transform=val_tfms,   root_dir=root_dir)
    return train_ds, val_ds

class RAFDBDatasetWithAUs(Dataset):
    def __init__(self, csv_path, transform=None, au_cols=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        
        if au_cols is None:
            au_cols = [c for c in self.df.columns if c.startswith("AU")]
        self.au_cols = au_cols

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        
        aus = torch.tensor(row[self.au_cols].values, dtype=torch.float32)

        
        aus = torch.clamp(aus, 0, 5) / 5.0  

        return image, aus, label

    def __len__(self):
        return len(self.df)