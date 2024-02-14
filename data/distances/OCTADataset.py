import os
import torch
import os.path as osp
import pandas as pd
import numpy as np 

from torch_geometric.data import Dataset, Data
from tqdm import tqdm


class OCTADataset(Dataset):
    def __init__(self, root='/media/data/anna_alex/OCTA-500_6mm/train_data', transform=None, pre_transform=None, pre_filter=None):
        img_folder = osp.join(root, 'raw')
        graphs_folder = osp.join(root, 'graphs')

        node_files = []
        edge_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            node_files.append(osp.join(graphs_folder, file_ + 'seg_nodes.csv'))
            edge_files.append(osp.join(graphs_folder, file_ + 'seg_edges.csv'))

        data_dicts = [
            {"nodes": node_file, "edges": edge_file} for
            node_file, edge_file in zip(node_files, edge_files)
        ]
        self.data = data_dicts
        self.name = 'real_octa'
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [d['edges'] for d in self.data]

    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx, _ in enumerate(self.data)]

    def download(self):
        pass

    def process(self):
        for idx in tqdm(range(len(self.raw_paths))):

            data = self.data[idx]
            nodes = pd.read_csv(data['nodes'], sep=";", index_col="id")
            edges = pd.read_csv(data['edges'], sep=";", index_col="id")
            
            nodes_pos = torch.tensor(nodes.to_numpy()[:, :2].astype(np.float32) / 128.)
            edges = torch.tensor(edges.to_numpy()[:, :2].astype(int)).t()

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