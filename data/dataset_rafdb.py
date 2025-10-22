# data/dataset_rafdb.py
from .dataset_base import CSVDataset

def build_rafdb(train_csv, val_csv, train_tfms, val_tfms, root_dir=None):
    train_ds = CSVDataset(train_csv, transform=train_tfms, root_dir=root_dir)
    val_ds   = CSVDataset(val_csv,   transform=val_tfms,   root_dir=root_dir)
    return train_ds, val_ds
