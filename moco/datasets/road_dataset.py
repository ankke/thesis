import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from PIL import Image


class MoCo_Road_Dataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, transform1, transform2):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform1 = transform1
        self.transform2 = transform2

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)

    def __getitem__(self, idx):
        image_data = Image.open(self.data[idx])

        image_data1 = self.transform1(image_data)
        image_data2 = self.transform2(image_data)

        image_data1 = tvf.normalize(image_data1, mean=self.mean, std=self.std)
        image_data2 = tvf.normalize(image_data2, mean=self.mean, std=self.std)

        return image_data1, image_data2


def build_moco_road_dataset(config, transform1, transform2, split=0.8, max_samples=0):
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    img_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))

    random.seed(config.DATA.SEED)
    random.shuffle(img_files)

    train_split = int(split*len(img_files))
    train_files, val_files = img_files[:train_split], img_files[train_split:]

    if max_samples > 0:
        train_files = train_files[:max_samples]
        val_files = val_files[:round(max_samples * (1 - split))]
        
    train_ds = MoCo_Road_Dataset(
        data=train_files,
        transform1=transform1,
        transform2=transform2
    )
    val_ds = MoCo_Road_Dataset(
        data=val_files,
        transform1=transform1,
        transform2=transform2
    )
    return train_ds, val_ds
