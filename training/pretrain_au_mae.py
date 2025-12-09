import os
import yaml
import torch
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.dataset_rafdb import RAFDBDatasetWithAUs
from data.dataloader_utils import build_loader
from data.transforms import get_train_transforms, get_val_transforms
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg):
    train_tfms = get_train_transforms(cfg["data"]["img_size"])
    val_tfms = get_val_transforms(cfg["data"]["img_size"])

    train_ds = RAFDBDatasetWithAUs(cfg["data"]["train_csv"], transform=train_tfms)
    val_ds = RAFDBDatasetWithAUs(cfg["data"]["val_csv"], transform=val_tfms)

    train_loader = build_loader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
    )
    val_loader = build_loader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return train_loader, val_loader, train_ds, val_ds


def save_reconstruction_grid(imgs, recons, epoch, out_dir, max_imgs=6):
    os.makedirs(out_dir, exist_ok=True)
    imgs = imgs[:max_imgs].cpu().permute(0, 2, 3, 1)
    recons = recons[:max_imgs].cpu().permute(0, 2, 3, 1)

    fig, axes = plt.subplots(2, max_imgs, figsize=(3 * max_imgs, 6))
    for i in range(max_imgs):
        axes[0, i].imshow(imgs[i].clamp(0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(recons[i].clamp(0, 1))
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"reconstruction_epoch{epoch}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def pretrain_mae(model, train_loader, val_loader, cfg, device):
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    num_epochs = cfg["training"]["epochs"]
    save_dir = cfg["training"].get("save_dir", "checkpoints_pretrain")
    vis_dir = os.path.join(save_dir, "reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for imgs, aus, _ in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}"):
            imgs, aus = imgs.to(device), aus.to(device)
            out = model(imgs, aus, labels=None, return_loss=True)
            loss = out["loss_recon"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Recon Loss: {avg_train_loss:.4f}")

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, aus, _ in val_loader:
                imgs, aus = imgs.to(device), aus.to(device)
                out = model(imgs, aus, labels=None, return_loss=True)
                val_loss += out["loss_recon"].item()

        val_loss /= len(val_loader)
        print(f"Val Recon Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"pretrain_epoch{epoch+1}.pth"))

        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            imgs_vis, aus_vis, _ = next(iter(val_loader))
            imgs_vis, aus_vis = imgs_vis.to(device), aus_vis.to(device)
            with torch.no_grad():
                out_vis = model(imgs_vis, aus_vis, labels=None, return_loss=False)
                recons = out_vis["recon_img"]
            save_reconstruction_grid(imgs_vis, recons, epoch + 1, vis_dir)


def pretrain_au_mae_main(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, val_loader, train_ds, _ = build_dataloaders(cfg)

    model_cfg = AUConditionedMAEConfig(
        image_size=cfg["data"]["img_size"],
        num_classes=None,
        conditioning=cfg["model"]["conditioning"],
        num_aus=len(train_ds.au_cols),
        mask_ratio=cfg["model"].get("mask_ratio", 0.75),
    )
    model = AUConditionedMAE(model_cfg).to(device)

    pretrain_mae(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    pretrain_au_mae_main()
