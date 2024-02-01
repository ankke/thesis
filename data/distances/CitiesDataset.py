import os
import torch
import pyvista

import os.path as osp
import numpy as np 

from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class CitiesDataset(Dataset):
    def __init__(self, root='/media/data/anna_alex/20cities/train_data', transform=None, pre_transform=None, pre_filter=None):
        img_folder = os.path.join(root, 'raw')
        vtk_folder = os.path.join(root, 'vtp')
        vtk_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))

        data_dicts = [
            {"vtp": vtk_file} for vtk_file in vtk_files
        ]
        self.data = data_dicts
        self.name = '20_cities'
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [d['vtp'] for d in self.data]

    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx, _ in enumerate(self.data)]

    def download(self):
        pass

    def process(self):
        for idx in tqdm(range(len(self.raw_paths))):

            data = self.data[idx]
            vtk_data = pyvista.read(data['vtp'])

            nodes = torch.tensor(np.float32(np.asarray(vtk_data.points)), dtype=torch.float)[:, :2]
            edges = torch.tensor(np.asarray(vtk_data.lines.reshape(-1, 3))[:, 1:], dtype=torch.int64).t()
                
            nodes_pos = nodes[:, :2]
            nodes_ones = torch.ones(nodes_pos.size(0)).view(-1, 1)

            data = Data(
                x=nodes_ones,
                edge_index=edges,
                pos=nodes_pos,
            )

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data