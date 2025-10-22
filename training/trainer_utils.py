import torch
def train_epoch(model, dataloader, optimizer, device, logger, epoch):
    model.train()
    total_loss = 0.0
    for i, (imgs, aus, labels) in enumerate(dataloader):
        imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
        out = model(imgs, aus, labels, return_loss=True)
        loss = out["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            msg = (f"[Epoch {epoch} | Batch {i}/{len(dataloader)}] "
                   f"Recon={out['loss_recon']:.4f}, "
                   f"Cls={out.get('loss_cls', torch.tensor(0.)).item():.4f}, "
                   f"Total={loss.item():.4f}")
            logger.write(msg)
    avg_loss = total_loss / len(dataloader)
    logger.write(f"Epoch {epoch} done. Avg loss={avg_loss:.4f}")
    return avg_loss


def validate_epoch(model, dataloader, device, logger, epoch):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, aus, labels in dataloader:
            imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
            out = model(imgs, aus, labels, return_loss=True)
            total_loss += out["loss_total"].item()
            if "logits" in out:
                preds = out["logits"].argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    acc = correct / total if total > 0 else 0
    logger.write(f"[Validation Epoch {epoch}] AvgLoss={avg_loss:.4f}, Accuracy={acc*100:.2f}%")
    return avg_loss, acc