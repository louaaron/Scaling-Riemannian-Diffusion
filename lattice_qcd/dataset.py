import torch
from sklearn.model_selection import train_test_split


def get_data(root, val):
    data = torch.load(root, map_location="cpu")
    train_data, val_data = train_test_split(data, test_size=val)
    return train_data, val_data