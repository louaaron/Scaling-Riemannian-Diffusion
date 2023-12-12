import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def to_xyz(latlong):
    latlong = latlong * np.pi / 180
    lat = latlong[:, 0]
    lon = latlong[:, 1]

    x = lat.cos() * lon.cos()
    y = lat.cos() * lon.sin()
    z = lat.sin()

    return torch.stack([x, y, z], dim=-1)

def get_data(cfg):
    assert cfg.data in ["fire", "flood", "quakes", "volc"]
    data = torch.load(f"{cfg.root}/{cfg.data}.pt")
    data = to_xyz(data)

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    other_data, test_data = train_test_split(data, test_size=test_ratio)
    train_data, val_data = train_test_split(other_data, test_size=val_ratio/(train_ratio+val_ratio))

    return train_data, val_data, test_data