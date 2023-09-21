import os
import numpy as np
import random
import torch
import pyvista
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as tvf
import pandas as pd
from utils.utils import rotate_coordinates
from torchvision.transforms.functional import rotate

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

    def __init__(self, data, augment, max_nodes):
        """[summary]

        Args:
            data ([type]): [description]
            augment ([type]): [description]
        """
        self.data = data
        self.augment = augment
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
        nodes = pd.read_csv(data['nodes'], sep=";", index_col="id")
        edges = pd.read_csv(data['edges'], sep=";", index_col="id")
        image_data = Image.open(data['img'])
        seg_data = Image.open(data['seg'])

        image_data = image_data.convert('RGB')
        image_data = torch.tensor(
            np.array(image_data), dtype=torch.float).permute(2, 0, 1)
        image_data = image_data/255.0

        seg_data = torch.tensor(
            np.array(seg_data), dtype=torch.float).unsqueeze(0)
        seg_data = seg_data / 255.0

        nodes = torch.tensor(nodes.to_numpy()[:, :2].astype(np.float32) / 128.)
        edges = torch.tensor(edges.to_numpy()[:, :2].astype(int))

        if self.augment:
            angle = random.randint(0, 3) * 90
            image_data = rotate(image_data, angle)
            seg_data = rotate(seg_data, angle)
            nodes = rotate_coordinates(nodes, angle)

        return image_data-0.5, seg_data-0.5, nodes, edges


def build_real_vessel_network_data(config, mode='train', split=0.95, max_samples=0, use_grayscale=False):
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
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        graphs_folder = os.path.join(config.DATA.DATA_PATH, 'graphs')

        img_files = []
        seg_files = []
        node_files = []
        edge_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + 'data.png'))
            seg_files.append(os.path.join(seg_folder, file_ + 'seg.png'))
            node_files.append(os.path.join(graphs_folder, file_ + 'seg_nodes.csv'))
            edge_files.append(os.path.join(graphs_folder, file_ + 'seg_edges.csv'))

        data_dicts = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files, seg_files, node_files, edge_files)
        ]
        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            augment=True,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return ds
    elif mode == 'test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        graphs_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'graphs')

        img_files = []
        seg_files = []
        node_files = []
        edge_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_ + 'data.png'))
            seg_files.append(os.path.join(seg_folder, file_ + 'seg.png'))
            node_files.append(os.path.join(graphs_folder, file_ + 'seg_nodes.csv'))
            edge_files.append(os.path.join(graphs_folder, file_ + 'seg_edges.csv'))

        data_dicts = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files, seg_files, node_files, edge_files)
        ]

        if max_samples > 0:
            data_dicts = data_dicts[:max_samples]

        ds = Vessel2GraphDataLoader(
            data=data_dicts,
            augment=False,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return ds
    elif mode == 'split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        graphs_folder = os.path.join(config.DATA.DATA_PATH, 'graphs')

        img_files = []
        seg_files = []
        node_files = []
        edge_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            seg_files.append(os.path.join(seg_folder, file_ + 'seg.png'))
            node_files.append(os.path.join(graphs_folder, file_+'seg_nodes.csv'))
            edge_files.append(os.path.join(graphs_folder, file_ + 'seg_edges.csv'))

        data_dicts = [
            {"img": img_file, "seg": seg_file, "nodes": node_file, "edges": edge_file} for
            img_file, seg_file, node_file, edge_file in zip(img_files, seg_files, node_files, edge_files)
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
            augment=True,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        val_ds = Vessel2GraphDataLoader(
            data=val_files,
            augment=False,
            max_nodes=config.MODEL.DECODER.OBJ_TOKEN
        )
        return train_ds, val_ds, None
