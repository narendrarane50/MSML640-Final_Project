# training/evaluate.py
import torch
from tqdm import tqdm
from .metrics import compute_metrics

@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    y_true, y_pred = [], []
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs  = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return compute_metrics(y_true, y_pred)
