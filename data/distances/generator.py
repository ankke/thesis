import os.path as osp
import torch
import os
from time import time
from torch_geometric.loader import DataLoader

from data.distances.ged import generate_ged
from data.distances.gmd import generate_gmd


RESULTS_DIR = '/media/data/anna_alex/distances/results'


def generate_random_pairs(method, dataset, num_pairs=100, **kwargs):
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # graph_pairs = [next(iter(data_loader)) for _ in range(num_pairs)]
    graph_pairs = [(dataset[i], dataset[i+1]) for i in range(0, num_pairs * 2, 2)]

    dir = osp.join(RESULTS_DIR, 'random_pairs', method)
    
    return generate(method, graph_pairs, dataset.name, dir, **kwargs)


def generate_one_to_many(method, dataset, id_, num_pairs=None, **kwargs):
    if num_pairs is None:
        num_pairs = len(dataset)

    query = dataset[id_]
    graph_pairs = [(query, graph) for graph in dataset[:num_pairs]]

    dir = osp.join(RESULTS_DIR, 'one_to_many', method, str(id_))
    
    return generate(method, graph_pairs, dataset.name, dir, **kwargs)

def generate(method, graph_pairs, dataset_name, dir, **kwargs):
    if not osp.exists(dir):
        os.makedirs(dir)

    meta_data = {
        'method': method,
        'dataset': dataset_name
    }

    if method == 'ged':
        timeout = kwargs.get('timeout', 10)
        return generate_ged(graph_pairs, dataset_name, dir, meta_data, timeout)
    
    elif method == 'gmd':
        C_V = kwargs.get('C_V', 1)
        C_E = kwargs.get('C_E', 1)
        M = kwargs.get('M', 1)

        return generate_gmd(graph_pairs, dataset_name, dir, meta_data, C_V=C_V, C_E=C_E, M=M)
    
    else:
        print(f'Choose available method: ged, gmd.')
