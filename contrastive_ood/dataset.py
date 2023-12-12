import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader


def get_data(root, id_data, dataset, dim=128):

    path = os.path.join(root, f"cider_outputs_{dim}", id_data, f"{dataset}.npy")
    data = np.load(path)
    data = torch.from_numpy(data)
    return data
