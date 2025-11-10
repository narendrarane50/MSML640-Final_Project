# training/ablation_runner.py
import os, yaml, torch
from torch import nn
from tqdm import tqdm
from data.transforms import get_train_transforms, get_val_transforms
from data.dataset_rafdb import build_rafdb
from data.dataloader_utils import build_loader
from models.fer_backbone import ResNet50Backbone, FERClassifier
from models.pose_normalizer import PoseNormalizer
from .evaluate import evaluate

def load_config(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def train_one_epoch(model, loader, optimizer, criterion, device="cuda"):
    model.train()
    running = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)

def run_ablation(cfg_path="config.yaml", dataset="rafdb"):
    cfg = load_config(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- datasets -----
    if dataset == "rafdb":
        train_ds, val_ds = build_rafdb(
            cfg["data"]["train_csv"], cfg["data"]["val_csv"],
            get_train_transforms(cfg["data"]["img_size"]),
            get_val_transforms(cfg["data"]["img_size"]),
            root_dir=cfg["data"].get("root_dir")
        )
    elif dataset == "fer2013":
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        data_cfg = cfg["fer2013"]
        train_dir = data_cfg["train_dir"]
        val_dir   = data_cfg["val_dir"]

        train_tfms = get_train_transforms(data_cfg["img_size"])
        val_tfms   = get_val_transforms(data_cfg["img_size"])

        print(f"\n[INFO] Loading FER-2013 from:\n  Train: {train_dir}\n  Val:   {val_dir}\n")

        train_ds = ImageFolder(root=train_dir, transform=train_tfms)
        val_ds   = ImageFolder(root=val_dir,   transform=val_tfms)
    else:
        raise ValueError("Implement other datasets similarly.")

    train_loader = build_loader(train_ds, cfg["train"]["batch_size"], True, cfg["train"]["num_workers"])
    val_loader   = build_loader(val_ds,   cfg["eval"]["batch_size"],  False, cfg["train"]["num_workers"])

    # ----- Ablation A: FER baseline (no pose normalization) -----
    print("\n[Ablation A] FER baseline (no pose normalization)")
    backbone = ResNet50Backbone(pretrained=True)
    model = FERClassifier(backbone, num_classes=cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["wd"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | loss={tr_loss:.4f}")
    metrics_A = evaluate(model, val_loader, device)
    print("Ablation A metrics:", metrics_A)

    # ----- Ablation B: FER + PoseNormalizer -----
    print("\n[Ablation B] FER + PoseNormalizer")
    backbone2 = ResNet50Backbone(pretrained=True)
    model2 = FERClassifier(backbone2, num_classes=cfg["model"]["num_classes"], use_pose_normalizer=True).to(device)

    # attach pose normalizer
    pose_norm = PoseNormalizer().to(device)
    model2.attach_pose_normalizer(pose_norm)

    # two learning rates: lower for FER backbone, slightly higher for PoseNormalizer
    optimizer2 = torch.optim.AdamW([
        # {"params": model2.pose_normalizer.parameters(), "lr": cfg["train"]["lr"] * 0.5},
        {"params": model2.pose_normalizer.parameters(), "lr": cfg["train"]["lr"] * 0.05},
        {"params": [p for n, p in model2.named_parameters() if "pose_normalizer" not in n], "lr": cfg["train"]["lr"] * 0.1}
    ], weight_decay=cfg["train"]["wd"])

    criterion = nn.CrossEntropyLoss()

    # --- Warmup: train PoseNormalizer standalone for 1 epoch ---
    print("\n[Warmup] Training PoseNormalizer only for 1 epoch")
    pose_optimizer = torch.optim.Adam(model2.pose_normalizer.parameters(), lr=1e-4)
    for imgs, _ in tqdm(train_loader, desc="Warmup PoseNormalizer"):
        imgs = imgs.to(device)
        pose_optimizer.zero_grad(set_to_none=True)
        warped = model2.pose_normalizer(imgs)
        # Identity-consistency loss (keeps transform near identity)
        loss = ((warped - imgs) ** 2).mean()
        loss.backward()
        pose_optimizer.step()
    print("[✓] Warmup complete — PoseNormalizer stabilized.\n")

    for epoch in range(cfg["train"]["epochs"]):
        # --- Phase 1: freeze FER backbone for first 2 epochs ---
        if epoch < 2:
            for name, param in model2.named_parameters():
                if "pose_normalizer" not in name:
                    param.requires_grad = False
        else:
            for param in model2.parameters():
                param.requires_grad = True

        tr_loss = train_one_epoch(model2, train_loader, optimizer2, criterion, device)
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | loss={tr_loss:.4f}")

    metrics_B = evaluate(model2, val_loader, device)
    print("Ablation B metrics:", metrics_B)

    return {"no_pose": metrics_A, "with_pose": metrics_B}

if __name__ == "__main__":
    run_ablation()
