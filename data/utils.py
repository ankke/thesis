import os

import networkx as nx
import matplotlib.pyplot as plt
from data.dataset_real_eye_vessels import Vessel2GraphOnlyDataLoader
from data.visualize_sample import draw_graph
import numpy as np
import os
import numpy as np

from PIL import Image
import torchvision.transforms.functional as tvf
import pandas as pd
import yaml
import json


def get_graph_data_loader(config):
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    graphs_folder = os.path.join(config.DATA.DATA_PATH, 'graphs')

    node_files = []
    edge_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        node_files.append(os.path.join(graphs_folder, file_ + 'seg_nodes.csv'))
        edge_files.append(os.path.join(graphs_folder, file_ + 'seg_edges.csv'))

    data_dicts = [
        {"nodes": node_file, "edges": edge_file} for
        node_file, edge_file in zip(node_files, edge_files)
    ]
    data_loader = Vessel2GraphOnlyDataLoader(
        data=data_dicts,
    )
    return data_loader

def draw_graphs_from_data_loader(config, samples_no=100):

    data_loader = get_graph_data_loader(config)
    
    fig, axs = plt.subplots(samples_no, 1)
    fig.set_size_inches(3, 4 * samples_no)

    for i in range(samples_no):
        nodes, edges = data_loader[i]
        nodes, edges = nodes.clone().cpu().detach(), edges.clone().cpu().detach()
        draw_graph(nodes, edges, axs[i])

    fig.canvas.draw()
    return fig 

def get_config(config_path):
    class obj:
        def __init__(self, dict1):
            self.__dict__.update(dict1)
            
    def dict2obj(dict1):
        return json.loads(json.dumps(dict1), object_hook=obj)
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2obj(config)

    return config