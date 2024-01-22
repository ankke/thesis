import os

import networkx as nx
import matplotlib.pyplot as plt
from data.dataset_real_eye_vessels import Vessel2GraphDataLoader
import numpy as np
import pandas as pd
from torchvision.transforms import Compose, Normalize
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


def draw_graph(nodes, edges, ax):
    ax.set_xlim(0,128)
    ax.set_ylim(0, 128)

    xs = nodes[:, 0] * 128.
    ys = nodes[:, 1] * 128.
    xs = 128. - xs
    ax.scatter(ys, xs)

    # Add all edges
    for edge in edges:
        ax.plot([ys[edge[0]], ys[edge[1]]], [xs[edge[0]], xs[edge[1]]], color="black")

def create_sample_visual(samples, number_samples=10):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    number_samples = min(number_samples, len(samples["images"]))
    fig, axs = plt.subplots(number_samples, 3, figsize=(1000 * px, number_samples * 300 * px))

    for i in range(number_samples):
        axs[i, 0].imshow(inv_norm(samples["images"][i].clone().cpu().detach()).permute(1, 2, 0))

        plt.sca(axs[i, 1])
        draw_graph(samples["nodes"][i].clone().cpu().detach(), samples["edges"][i].clone().cpu().detach(), axs[i, 1])
        plt.sca(axs[i, 2])
        draw_graph(samples["pred_nodes"][i].clone().cpu().detach(), samples["pred_edges"][i], axs[i, 2])

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).copy()
    data_1 = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    res = np.expand_dims(np.transpose(data_1, (2, 0, 1)), axis=0)
    return  res 


def create_sample_visual_from_data_loader(data_loader, samples_no=10, augment=False):
    data_loader.augment = augment
    
    fig, axs = plt.subplots(samples_no, 3)
    fig.set_size_inches(20,6 * samples_no)

    for i in range(samples_no):
        image, seg, nodes, edges, _ = data_loader[i]
        image, seg, nodes, edges = image.clone().cpu().detach(), seg.clone().cpu().detach(), nodes.clone().cpu().detach(), edges.clone().cpu().detach()
        axs[i, 0].imshow(inv_norm(image).permute(1, 2, 0))
        seg = torch.reshape(seg, (128,128))
        axs[i, 1].imshow(seg, cmap='gray' )
        draw_graph(nodes, edges, axs[i, 2])

    fig.canvas.draw()
    return fig 

inv_norm = Compose([
    Normalize(
        mean=[0, 0, 0],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1, 1, 1]
    ),
])

