

import os
import yaml
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader_utils import build_loader
from data.dataset_rafdb import build_rafdb
from data.transforms import get_train_transforms, get_val_transforms
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig
from training.metrics import compute_metrics
from utils.logging_utils import Logger


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    train_tfms = get_train_transforms(cfg["data"]["img_size"])
    val_tfms = get_val_transforms(cfg["data"]["img_size"])

    train_ds, val_ds = build_rafdb(
            cfg["data"]["train_csv"], cfg["data"]["val_csv"],
            get_train_transforms(cfg["data"]["img_size"]),
            get_val_transforms(cfg["data"]["img_size"]),
            root_dir=cfg["data"].get("root_dir")
        )
    
    train_loader = build_loader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader = build_loader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )
    num_classes = len(set([lbl for _, lbl in train_ds.samples]))
    return train_loader, val_loader, num_classes


def train_mae_baseline(model, train_loader, val_loader, cfg, device, logger):
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-4),
    )
    num_epochs = cfg["train"]["epochs"]
    save_dir = cfg["train"].get("save_dir", "checkpoints_baseline")
    os.makedirs(save_dir, exist_ok=True)

    
    log_path = os.path.join(save_dir, "metrics_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "epoch", "train_loss", "val_loss", "accuracy", "balanced_accuracy", "macro_f1"
        ]).to_csv(log_path, index=False)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs, aus=None, labels=labels)
            loss = out["loss_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs, aus=None, labels=labels)
                val_loss += out["loss_total"].item()
                preds = torch.argmax(out["logits"], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

        
        msg = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )
        logger.write(msg)
        logger.write(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        
        pd.DataFrame([{
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        }]).to_csv(log_path, mode="a", index=False, header=False)

        
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{epoch+1}.pth"))

    logger.write(f"\n[✔] Training complete. Metrics logged to {log_path}")
    logger.close()



def main():
    cfg = load_config("config.yaml")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(cfg)


    logger = Logger(log_dir="logs", filename="baseline_training.log")


    model_cfg = AUConditionedMAEConfig(
        image_size=cfg["data"]["img_size"],
        num_classes=num_classes,
        conditioning="none",
        num_aus=0,
        mask_ratio=cfg["au_model"].get("mask_ratio", 0.75),
        patch_size=cfg["au_model"].get("patch_size", 16),
        decoder_embed_dim=cfg["au_model"].get("decoder_embed_dim", 512),
        decoder_depth=cfg["au_model"].get("decoder_depth", 8),
        decoder_num_heads=cfg["au_model"].get("decoder_num_heads", 16),
        decoder_mlp_ratio=cfg["au_model"].get("decoder_mlp_ratio", 4.0),
        in_chans=cfg["au_model"].get("in_chans", 3)
    )
    model = AUConditionedMAE(model_cfg).to(device)

    train_mae_baseline(model, train_loader, val_loader, cfg, device, logger)


if __name__ == "__main__":
    main()
