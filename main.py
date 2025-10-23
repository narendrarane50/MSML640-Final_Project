# main.py
import argparse
from training.ablation_runner import run_ablation

from torch.utils.data import DataLoader, TensorDataset
import torch
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedMAEConfig
from utils.logging_utils import Logger
from training.trainer_utils import train_epoch, validate_epoch

def run_training_example():
    
    # Mock dataset
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--dataset", type=str, default="rafdb", choices=["rafdb","fer2013","affectnet"])
    args = ap.parse_args()

    results = run_ablation(cfg_path=args.config, dataset=args.dataset)
    print("\n=== Ablation Summary ===")
    for k, v in results.items():
        print(k, {m: (round(x,4) if m!="confusion_matrix" else "\n"+str(x)) for m, x in v.items()})
    
    run_training_example()
