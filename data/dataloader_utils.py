
from torch.utils.data import DataLoader

def build_loader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=pin_memory)
