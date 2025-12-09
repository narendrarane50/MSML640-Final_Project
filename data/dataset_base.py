
import os, csv
from PIL import Image
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    CSV format: path,label
    label in [0..6] for 7 FER classes (adjust if your mapping differs).
    """
    def __init__(self, csv_file, transform=None, root_dir=None):
        self.samples = []
        self.transform = transform
        self.root_dir = root_dir
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2: continue
                path, label = row[0], int(row[1])
                if root_dir and not os.path.isabs(path):
                    path = os.path.join(root_dir, path)
                self.samples.append((path, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)
