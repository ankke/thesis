import os
import numpy as np
import torch
from torch.utils.data import Dataset
from moco.loader import GaussianBlur, Solarize
import torchvision.transforms as transforms
from PIL import Image
import random
import torchvision.transforms.functional as TF

class DiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class MoCo_Real_Eye_Dataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, img_size):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.)),
            DiscreteRotation([0,90,180, 270]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.5, 1.)),
            DiscreteRotation([0,90,180, 270]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)


    def __getitem__(self, idx):
        image_data = Image.open(self.data[idx])

        image_data = image_data.convert('RGB')

        image_data1 = self.transform1(image_data)
        image_data2 = self.transform2(image_data)

        return (image_data1/255.0) - 0.5, (image_data2/255.0) - 0.5


def build_moco_real_eye_dataset(config, max_samples=0):
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    img_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))

    if max_samples > 0:
        img_files = img_files[:max_samples]
        
    train_ds = MoCo_Real_Eye_Dataset(
        data=img_files,
        img_size=config.DATA.IMG_SIZE,
    )
    return train_ds
