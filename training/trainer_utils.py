import torch

def train_epoch(model, dataloader, optimizer, device, logger, epoch):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)

    for imgs, aus, labels in dataloader:
        imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
        out = model(imgs, aus, labels, return_loss=True)
        loss = out["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    logger.write(f"[Train Epoch {epoch}] Avg Loss={avg_loss:.4f}")

    return avg_loss


def validate_epoch(model, dataloader, device, logger, epoch):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, aus, labels in dataloader:
            imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
            out = model(imgs, aus, labels, return_loss=True)

            running_loss += out["loss_total"].item()

            if "logits" in out:
                preds = out["logits"].argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    acc = (correct / total * 100) if total > 0 else 0.0

    logger.write(f"[Validation Epoch {epoch}] AvgLoss={avg_loss:.4f}, Accuracy={acc:.2f}%")

    return avg_loss, acc
