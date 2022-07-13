# Fns to take np image dataset and dataframe, and
# convert to a dataset amenable to train torch models

import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torchvision.io import read_image


def make_torch_dataloader(images, labels, bs=64, shuffle=False):
    data = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    data_loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=shuffle)
    return data_loader
