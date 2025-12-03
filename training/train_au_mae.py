import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torchvision import transforms
from utils.yaml_loader import load_yaml_config
# -------------------------
# Imports
# -------------------------
from models.au_conditioned_mae import AUConditionedMAE
from data.dataset_rafdb import RAFDB_AU_Dataset

# -------------------------
# CONFIG
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

img_dir = load_yaml_config("config_aus.yaml",'image_directory')
splits = load_yaml_config("config_aus.yaml",'au_splits')
train_csv = splits['aus_training_split']
test_csv  = splits['aus_validation_split']
train_root = img_dir['train_images_dir']
test_root  = img_dir['test_images_dir']

epochs = 15
batch_size = 32
lr = 1e-4
num_classes = 7  # RAF-DB emotions

os.makedirs("Models", exist_ok=True)

# -------------------------
# TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# DATASETS
# -------------------------
train_data = RAFDB_AU_Dataset(train_csv, train_root, transform)
test_data  = RAFDB_AU_Dataset(test_csv, test_root, transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

au_dim = len([c for c in pd.read_csv(train_csv).columns if "AU" in c])
print(f"[INFO] AU feature dimension: {au_dim}")

# -------------------------
# MODEL
# -------------------------
model = AUConditionedMAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

print(f"[INFO] AUConditionedMAE initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

# -------------------------
# TRAINING LOOP
# -------------------------
train_losses, test_losses = [], []
best_f1, best_acc = 0.0, 0.0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for imgs, aus, _ in train_loader:
        imgs, aus = imgs.to(device), aus.to(device)
        preds = model(imgs, aus)
        loss = criterion(preds, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_loss:.4f}")

    # -------------------------
    # EVALUATION
    # -------------------------
    model.eval()
    y_true, y_pred = [], []
    test_loss = 0.0

    with torch.no_grad():
        for imgs, aus, labels in test_loader:
            imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
            recon = model(imgs, aus)
            loss = criterion(recon, imgs)
            test_loss += loss.item()

            # Compute pseudo emotion label prediction from AU correlation
            preds = (recon.mean(dim=[1, 2, 3]) > 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    saved_models_dir = load_yaml_config("config_aus.yaml",'au_mae_model_path')
    print(f"  → Val Loss: {test_loss:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")
    os.makedirs("Saved_Models", exist_ok=True)
    if f1 > best_f1:
        best_f1, best_acc = f1, acc
        torch.save(model.state_dict(), saved_models_dir['pretrained_au_mae_weights'])
        torch.save(model, saved_models_dir['pretrained_au_mae_model'])
        print(f"  ✅ Best model saved (F1={f1:.3f}, Acc={acc:.3f})")

# -------------------------
# PLOTS
# -------------------------

os.makedirs("Plots", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("AUConditionedMAE Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/au_mae_loss_curve.png")
plt.show()

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - AUConditionedMAE (Test Split)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

plt.savefig("Plots/au_mae_confusion_matrix.png")
plt.show()

print(f"[DONE] Training completed. Best F1: {best_f1:.3f}, Acc: {best_acc:.3f}")
