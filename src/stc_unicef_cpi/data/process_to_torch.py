import glob
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset

from .process_geotiff import extract_ims_from_hex_codes


def make_torch_dataloader_from_numpy(images, labels, bs=64, shuffle=False):
    """Take np image dataset and dataframe, and
    convert to a dataset amenable to train torch models


    :param images: _description_
    :type images: _type_
    :param labels: _description_
    :type labels: _type_
    :param bs: _description_, defaults to 64
    :type bs: int, optional
    :param shuffle: _description_, defaults to False
    :type shuffle: bool, optional
    :return: _description_
    :rtype: _type_
    """
    data = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    data_loader = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=shuffle)
    return data_loader


class HexDataset(Dataset):
    """Make a torch dataset that constructs images from
    tiff files according to hex codes

    :param Dataset: _description_
    :type Dataset: _type_
    """

    def __init__(
        self,
        tiff_dir,
        hex_codes,
        labels,
        width=33,
        height=33,
        transform=None,
        target_transform=None,
    ):
        # absolute path to search for all tiff files inside a specified folder
        path = str(Path(tiff_dir) / "*.tif")
        self.tif_files = glob.glob(path)
        try:
            assert len(hex_codes) == len(labels)
        except AssertionError:
            raise ValueError("Hex codes and labels must be the same length")
        self.labels = labels
        self.hex_codes = hex_codes
        self.transform = transform
        self.target_transform = target_transform
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hex_code = self.hex_codes[idx]
        image = extract_ims_from_hex_codes(
            self.tif_files, [hex_code], width=self.width, height=self.height
        )
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
