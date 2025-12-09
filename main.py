import argparse
from training.ablation_runner import run_ablation
from torch.utils.data import DataLoader, TensorDataset
import torch
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig
from utils.logging_utils import Logger
from training.trainer_utils import train_epoch, validate_epoch
from data.dataset_rafdb_au import RAFDB_AU_Dataset
from data.transforms import get_train_transforms, get_val_transforms

def run_training_example():
    
    B, N = 32, 100

    ds = TensorDataset(imgs, aus, labels)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AUConditionedMAEConfig(conditioning="both")
    model = AUConditionedMAE(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logger = Logger()

    for epoch in range(1, 3):
        train_epoch(model, loader, optim, device, logger, epoch)
        validate_epoch(model, loader, device, logger, epoch)

    logger.close()




def run_au_mae_pretrain(cfg_path):
    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    train_ds = RAFDB_AU_Dataset(
        csv_path="splits/aus_raf_train.csv",
        root_dir=cfg["data"]["root_dir"],
        transform=get_train_transforms(cfg["data"]["img_size"]),
        split="train",
    )

    val_ds = RAFDB_AU_Dataset(
        csv_path="splits/aus_raf_test.csv",
        root_dir=cfg["data"]["root_dir"],
        transform=get_val_transforms(cfg["data"]["img_size"]),
        split="test",
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

    print(f"[INFO] Loaded RAF-DB AU dataset: {len(train_ds)} training samples")

    mae_cfg = AUConditionedMAEConfig(conditioning="both")
    model = AUConditionedMAE(mae_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    logger = Logger()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        print(f"\n[AU-MAE] Epoch {epoch}")
        train_epoch(model, train_loader, optimizer, device, logger, epoch)
        validate_epoch(model, val_loader, device, logger, epoch)

    torch.save(model.state_dict(), "au_mae_pretrained.pth")
    print("\n[✓] AU-MAE pretraining complete. Saved: au_mae_pretrained.pth\n")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--dataset", type=str, default="rafdb", choices=["rafdb","fer2013","affectnet"])
    ap.add_argument("--mode", type=str, default="ablation",
                choices=["ablation", "pretrain_au_mae"],
                help="Choose ablation or AU-MAE pretraining")

    args = ap.parse_args()

    print(f"\n[INFO] Using dataset: {args.dataset}")
    print(f"[INFO] Loading configuration from: {args.config}\n")
    
    if args.mode == "pretrain_au_mae":
        run_au_mae_pretrain(args.config)
        exit()


    results = run_ablation(cfg_path=args.config, dataset=args.dataset)
    print("\n=== Ablation Summary ===")
    for k, v in results.items():
        print(k, {m: (round(x,4) if m!="confusion_matrix" else "\n"+str(x)) for m, x in v.items()})
    
    # run_training_example()
