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
# --- Ablation C imports ---
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig
from data.dataset_rafdb_au import RAFDB_AU_Dataset


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

    # ============================================
    # Ablation C — AU-MAE pretrained → Fine-tuned FER
    # ============================================
    print("\n[Ablation C] AU-MAE pretrained → Fine-tuned on RAF-DB")

    # 1) Load AU-MAE model + pretrained weights
    mae_cfg = AUConditionedMAEConfig(conditioning="both")
    modelC = AUConditionedMAE(mae_cfg).to(device)

    if not os.path.exists("au_mae_pretrained.pth"):
        raise FileNotFoundError("You must run AU-MAE pretraining first (mode=pretrain_au_mae).")

    print("[Ablation C] Loading: au_mae_pretrained.pth")
    modelC.load_state_dict(torch.load("au_mae_pretrained.pth", map_location=device))

    # # 2) Freeze encoder blocks 0–7
    # for i, blk in enumerate(modelC.encoder.blocks):
    #     if i < 8:
    #         for p in blk.parameters():
    #             p.requires_grad = False

    # # 3) Unfreeze blocks 8–11 + classifier
    # for i, blk in enumerate(modelC.encoder.blocks):
    #     if i >= 8:
    #         for p in blk.parameters():
    #             p.requires_grad = True

    # for p in modelC.classifier.parameters():
    #     p.requires_grad = True

    print("[Ablation C] Freezing encoder blocks 0–5, training blocks 6–11")

    # Freeze all parameters first
    for param in modelC.parameters():
        param.requires_grad = False

    # Unfreeze last 6 encoder blocks (6,7,8,9,10,11)
    trainable_blocks = list(range(6, 12))

    for idx in trainable_blocks:
        for name, param in modelC.encoder.blocks[idx].named_parameters():
            param.requires_grad = True

    # Also unfreeze classifier head
    for param in modelC.classifier.parameters():
        param.requires_grad = True

    # 4) Build RAF-DB AU dataset
    train_dsC = RAFDB_AU_Dataset(
        csv_path=cfg["data"]["au_train_csv"],
        root_dir=cfg["data"]["root_dir"],
        transform=get_train_transforms(cfg["data"]["img_size"]),
        split="train",
    )
    val_dsC = RAFDB_AU_Dataset(
        csv_path=cfg["data"]["au_val_csv"],
        root_dir=cfg["data"]["root_dir"],
        transform=get_val_transforms(cfg["data"]["img_size"]),
        split="test",
    )

    train_loaderC = build_loader(train_dsC, cfg["train"]["batch_size"], True, cfg["train"]["num_workers"])
    val_loaderC   = build_loader(val_dsC,   cfg["eval"]["batch_size"],  False, cfg["train"]["num_workers"])

    # 5) Optimizer — smaller LR for encoder, full LR for classifier + last 6 blocks
    enc_params = []
    head_params = []

    for name, param in modelC.named_parameters():
        if not param.requires_grad:
            continue

        # classifier: high LR
        if "classifier" in name:
            head_params.append(param)

        # last 6 blocks: high LR
        elif any(f"encoder.blocks.{i}" in name for i in range(6, 12)):
            head_params.append(param)

        # everything else (unfrozen parts)
        else:
            enc_params.append(param)

    optimizerC = torch.optim.AdamW([
        {"params": enc_params,  "lr": cfg["train"]["lr"] * 0.1},
        {"params": head_params, "lr": cfg["train"]["lr"]},
    ], weight_decay=cfg["train"]["wd"])

    # 6) Train 10 epochs
    criterionC = nn.CrossEntropyLoss()
    for epoch in range(cfg["train"]["epochs"]):
        modelC.train()
        running = 0.0
        for imgs, aus, labels in train_loaderC:
            imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
            optimizerC.zero_grad()
            out = modelC(imgs, aus, labels, return_loss=True)
            loss = out["loss_total"]
            loss.backward()
            optimizerC.step()
            running += loss.item() * imgs.size(0)
        print(f"[Ablation C] Epoch {epoch+1} | loss={running/len(train_loaderC.dataset):.4f}")

    # 7) Final evaluation
    def eval_wrapper(model, loader, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, aus, labels in loader:  # <--- AUS INCLUDED
                imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)

                # use AUs during forward pass
                out = model(imgs, aus, labels=None, return_loss=False)

                preds = out["logits"].argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        return {"accuracy": acc}

    metrics_C = eval_wrapper(modelC, val_loaderC, device)

    print("Ablation C metrics:", metrics_C)


    return {
        "no_pose": metrics_A,
        "with_pose": metrics_B,
        "au_mae_finetune": metrics_C
    }

    # return {"au_mae_finetune": metrics_C}



if __name__ == "__main__":
    run_ablation()
