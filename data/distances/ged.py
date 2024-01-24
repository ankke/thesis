import networkx as nx
import os.path as osp
import torch

from time import time
from multiprocessing import Pool, cpu_count
from torch_geometric.utils import to_networkx


NUM_PROCESSES = cpu_count()


def _compute_ged(args):
    data_pair, timeout = args
    graph1 = to_networkx(data_pair[0]).to_undirected()
    graph2 = to_networkx(data_pair[1]).to_undirected()
    distance = nx.graph_edit_distance(graph1, graph2, timeout=timeout)
    return {'pair': data_pair, 'ged': distance}


def _ged(graph_pairs, timeout=30):
    args = [(graph_pair, timeout) for graph_pair in graph_pairs]
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(_compute_ged, args, chunksize=1)
    return results


def generate_ged(graph_pairs, dataset_name, dir, meta_data, timeout):
    num_pairs = len(graph_pairs)

    file_name = f'{dataset_name}_t{timeout}_n{num_pairs}.pth'
    file_path = osp.join(dir, file_name)

    start_time = time()

    ged_res = _ged(graph_pairs, timeout=timeout)

    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.1f} seconds")

    meta_data['timeout'] = timeout
    torch.save((ged_res, meta_data), file_path)

    return file_path


