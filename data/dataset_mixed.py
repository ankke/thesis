import os
import numpy as np
import random
import imageio
import torch
import pyvista
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms.functional as tvf
from PIL import Image

from data.dataset_road_network import build_road_network_data

def build_mixed_data(config, mode='split', split=0.95, max_samples=0, use_grayscale=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    config.DATA.DATA_PATH = config.DATA.SOURCE_DATA_PATH
    source_train_data, source_val_data = build_road_network_data(config, mode, split, max_samples, use_grayscale, domain_classification=0)
    config.DATA.DATA_PATH = config.DATA.TARGET_DATA_PATH
    target_train_data, target_val_data = build_road_network_data(config, mode, split, max_samples, use_grayscale, domain_classification=1)

    train_ds = ConcatDataset([source_train_data, target_train_data])
    val_ds = ConcatDataset([source_val_data, target_val_data])

    return train_ds, val_ds

    
