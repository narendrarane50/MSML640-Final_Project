import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data.dataset_rafdb import RAFDB_AU_Dataset
from models.au_conditioned_mae import AUConditionedMAE, AUConditionedFER
from utils.yaml_loader import load_yaml_config 

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img_dir = load_yaml_config("config_aus.yaml",'image_directory')
splits = load_yaml_config("config_aus.yaml",'au_splits')
train_csv = splits['aus_training_split']
test_csv  = splits['aus_validation_split']
train_root = img_dir['train_images_dir']
test_root  = img_dir['test_images_dir']

train_data = RAFDB_AU_Dataset(train_csv, train_root, transform)
test_data  = RAFDB_AU_Dataset(test_csv, test_root, transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

# --- Load pretrained MAE ---
pretrained_mae = torch.load("Models/model_1.pkl", map_location=device)

model = AUConditionedFER(pretrained_mae, num_classes=7).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10

# --- Training ---
for epoch in range(epochs):
    model.train()
    train_loss, correct = 0, 0
    for imgs, aus, labels in train_loader:
        imgs, aus, labels = imgs.to(device), aus.to(device), labels.to(device)
        logits = model(imgs, aus)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()

    acc = correct / len(train_data)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.3f}")

# --- Evaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, aus, labels in val_loader:
        imgs, aus = imgs.to(device), aus.to(device)
        preds = model(imgs, aus).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)
print(f"\nValidation Accuracy: {acc:.3f} | Macro F1: {f1:.3f}")

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - AUConditionedFER (Pose Normalized)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_pose_normalized.png")
plt.show()
