# training/train_au_mae.py

from torch.utils.data import DataLoader
from data.dataset_rafdb_au import RAFDB_AU_Dataset
from data.transforms import get_train_transforms, get_val_transforms
import torch
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig

def get_rafdb_au_loaders(cfg):
    train_ds = RAFDB_AU_Dataset(
        csv_path=cfg["data"]["train_csv"],
        root_dir=cfg["data"]["root_dir"],
        transform=get_train_transforms(cfg["data"]["img_size"])
    )

    val_ds = RAFDB_AU_Dataset(
        csv_path=cfg["data"]["val_csv"],
        root_dir=cfg["data"]["root_dir"],
        transform=get_val_transforms(cfg["data"]["img_size"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"]
    )

    return train_loader, val_loader


def train_au_mae(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_rafdb_au_loaders(cfg)

    mae_cfg = AUConditionedMAEConfig(conditioning="both")
    model = AUConditionedMAE(mae_cfg).to(device)

    # TODO: add your AU-MAE training loop here:
    #  - forward with masking
    #  - reconstruction loss + AU loss
    #  - optimizer / epochs etc.

    print("[INFO] AU-MAE training stub ready (data pipeline works).")
