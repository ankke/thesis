import os
import numpy as np
import random
import torch
import pyvista
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as tvf

# train_transform = Compose(
#     [
#         Flip,
#         Rotate90,
#         ToTensor,
#     ]
# )
train_transform = []
# val_transform = Compose(
#     [
#         ToTensor,
#     ]
# )
val_transform = []


class Vessel2GraphDataLoader(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, transform, max_nodes):
        """[summary]

        Args:
            data ([type]): [description]
            transform ([type]): [description]
        """
        self.data = data
        self.transform = transform
        self.max_nodes = max_nodes

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.data)

    def __getitem__(self, idx):
        """[summary]

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = self.data[idx]
        vtk_data = pyvista.read(data['vtp'])
        raw_data = Image.open(data['img'])

        seg_data = torch.tensor(np.array(raw_data)).unsqueeze(0) / 255.

        image_data = raw_data.convert('RGB')
        image_data = torch.tensor(
            np.array(image_data), dtype=torch.float).permute(2, 0, 1)
        image_data = image_data/255.0

        coordinates = torch.tensor(np.float32(
            np.asarray(vtk_data.points)), dtype=torch.float)
        lines = torch.tensor(np.asarray(
            vtk_data.lines.reshape(-1, 3)), dtype=torch.int64)

        lines = lines[:, 1:]

        if coordinates[:, :2].shape[0] > self.max_nodes:
            print("Warning: Too many nodes in sample. Nodes will be deleted")
            lines = lines[lines[:, 0] < self.max_nodes]
            lines = lines[lines[:, 1] < self.max_nodes]


        return image_data-0.5, seg_data-0.5, coordinates[:self.max_nodes, :2], lines


def build_synthetic_vessel_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """
    if mode == 'train':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file} for img_file, vtk_file in zip(img_files, vtk_files)
        ]
        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return ds
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file} for img_file, vtk_file in zip(img_files, vtk_files)
        ]

        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]

        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return ds
    elif mode == 'split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file} for img_file, vtk_file in zip(img_files, vtk_files)
        ]
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:
                                            train_split], data_dicts[train_split:]

        if max_samples > 0:
            train_files = train_files[:max_samples]
            val_files = val_files[:round(max_samples * (1 - split))]

        train_ds = Vessel2GraphDataLoader(
            data=train_files,
            transform=train_transform,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        val_ds = Vessel2GraphDataLoader(
            data=val_files,
            transform=val_transform,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return train_ds, val_ds
