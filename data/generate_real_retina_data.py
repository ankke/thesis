import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
import json
from argparse import ArgumentParser
import networkx as nx
import pandas as pd
from PIL import Image
import re
from tqdm import tqdm

patch_size = [128, 128, 1]
pad = [20, 20, 0]

total_min = 1
total_max = 0

def save_input(path, idx, patch, patch_seg, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """

    global total_min
    global total_max
    imageio.imwrite(path+'raw/sample_'+str(idx).zfill(6)+'_data.png', patch)
    imageio.imwrite(path+'seg/sample_'+str(idx).zfill(6)+'_seg.png', patch_seg)

    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)

    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'mesh/sample_'+str(idx).zfill(4)+'_segmentation.stl')

    patch_edge = np.concatenate(
        (np.int32(2*np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    cur_min = np.min(patch_coord[:, :2])
    cur_max = np.max(patch_coord[:, :2])
    if cur_min < total_min:
        total_min = cur_min
        print(total_min)
    if cur_max > total_max:
        total_max = cur_max
        print(total_max)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image, seg,  mesh, device=None):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    global image_id
    p_h, p_w, _ = patch_size
    pad_h, pad_w, _ = pad

    p_h = p_h - 2*pad_h
    p_w = p_w - 2*pad_w

    h, w = image.shape
    x_ = np.int32(np.linspace(5, h-5-p_h, 8))
    y_ = np.int32(np.linspace(5, w-5-p_w, 8))

    ind = np.meshgrid(x_, y_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(2, -1).T)):
        # print(image.shape, seg.shape)
        start = np.array((start[0], start[1], 0))
        end = start + np.array(patch_size)-1 - 2*np.array(pad)

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1] + p_w], ((pad_h, pad_h), (pad_w, pad_w)))
        patch_list = [patch]

        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w], ((pad_h, pad_h), (pad_w, pad_w)))
        seg_list = [patch_seg]

        # collect all the nodes
        bounds = [start[0], end[0], start[1], end[1], -0.5, 0.5]

        clipped_mesh = mesh.clip_box(bounds, invert=False)
        patch_coordinates = np.float32(np.asarray(clipped_mesh.points))
        patch_edge = clipped_mesh.cells[np.sum(
            clipped_mesh.celltypes == 1)*2:].reshape(-1, 3)

        patch_coord_ind = np.where(
            (np.prod(patch_coordinates >= start, 1)*np.prod(patch_coordinates <= end, 1)) > 0.0)
        # all coordinates inside the patch
        patch_coordinates = patch_coordinates[patch_coord_ind[0], :]
        patch_edge = [tuple(l) for l in patch_edge[:, 1:] if l[0]
                      in patch_coord_ind[0] and l[1] in patch_coord_ind[0]]

        # flatten all the indices of the edges which completely lie inside patch
        temp = np.array(patch_edge).flatten()
        # remap the edge indices according to the new order
        temp = [np.where(patch_coord_ind[0] == ind) for ind in temp]
        # reshape the edge list into previous format
        patch_edge = np.array(temp).reshape(-1, 2)

        if patch_coordinates.shape[0] < 2 or patch_edge.shape[0] < 1 or patch_coordinates.shape[0] > 80:
            continue
        # concatenate final variables
        patch_coordinates = (patch_coordinates-start +
                             np.array(pad))/np.array(patch_size)
        patch_coord_list = [patch_coordinates]  # .to(device))
        patch_edge_list = [patch_edge]  # .to(device))

        mod_patch_coord_list, mod_patch_edge_list = (
            patch_coord_list, patch_edge_list)

        # save data
        for patch, patch_seg, patch_coord, patch_edge in zip(patch_list, seg_list, mod_patch_coord_list, mod_patch_edge_list):
            if patch_seg.sum() > 10:
                save_input(save_path, image_id, patch,
                           patch_seg, patch_coord, patch_edge)
                image_id = image_id+1
                # print('Image No', image_id)

def create_graph(node_path, edge_path):
    nodes = pd.read_csv(node_path, sep=";", index_col="id")
    edges = pd.read_csv(edge_path, sep=";", index_col="id")
    max_node_id = len(nodes)
    G = nx.MultiGraph()

    for idxN, node in nodes.iterrows():
        G.add_node(idxN, pos=(int(node["pos_x"]), int(node["pos_y"]), int(node["pos_z"])))
    # add the edges
    for idxE, edge in edges.iterrows():
        G.add_edge(edge["node1id"], edge["node2id"])
    return G

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    required=True,
                    help='config file (.yml) containing the hyper-parameters for data. '
                         'Should be the same as for training/testing. See /config for examples.')
parser.add_argument('--source',
                    default=None,
                    required=True,
                    help='Path to source directory')
parser.add_argument('--target',
                    default=None,
                    required=True,
                    help='Path to target directory')
parser.add_argument('--split',
                    default=0.8,
                    type=float,
                    help='Train/Test split. 0.8 means 80% of the data will be training data and 20% testing data'
                        'Default: 0.8')
parser.add_argument('--city_names',
                    default=None,
                    required=False,
                    help='Path to json with city names that are prefixing the raw source images')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    required=False,
                    help='Random seed')

image_id = 1

def generate_data(args):
    global image_id

    root_dir = args.source
    target_dir = args.target

    img_dir = os.path.join(root_dir, 'img')
    seg_dir = os.path.join(root_dir, 'seg')
    graph_dir = os.path.join(root_dir, 'graphs')

    raw_files = []
    seg_files = []
    nodes_files = []
    edges_files = []

    png_regex = re.compile(r'.*\.png')

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if png_regex.match(file):
                file_id = file[:-4]
                raw_files.append(os.path.join(img_dir, f'{file_id}.png'))
                seg_files.append(os.path.join(seg_dir, f'{file_id}.png'))
                nodes_files.append(os.path.join(graph_dir, f'{file_id}_nodes.csv'))
                edges_files.append(os.path.join(graph_dir, f'{file_id}_edges.csv'))

    # Sets the seed for reproducibility
    random.seed(args.seed)

    split = round(args.split * len(raw_files))

    train_path = f"{target_dir}/train_data/"
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path + '/seg')
        os.makedirs(train_path + '/vtp')
        os.makedirs(train_path + '/raw')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')

    for raw_file, seg_file, nodes_file, edges_file in tqdm(zip(raw_files[:split], seg_files[:split], nodes_files[:split], edges_files[:split])):
        raw_image = np.array(Image.open(raw_file))
        seg_image = np.array(Image.open(seg_file))
        graph = create_graph(nodes_file, edges_file)

        nodes = np.array([pos for node_id, pos in graph.nodes(data="pos")])
        nodes[:, 2] = 0

        edges = np.array(graph.edges, dtype=np.int32)
        edges = edges[:, [2, 0, 1]]
        edges[:, 0] = 2

        mesh = pyvista.PolyData(nodes)
        mesh.lines = edges.flatten()

        patch_extract(train_path, raw_image, seg_image, mesh)

    image_id = 1
    test_path = f"{target_dir}/test_data/"
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path + '/seg')
        os.makedirs(test_path + '/vtp')
        os.makedirs(test_path + '/raw')
    else:
        raise Exception("Test folder is non-empty")

    print('Preparing Test Data')

    for raw_file, seg_file, nodes_file, edges_file in tqdm(
            zip(raw_files[split:], seg_files[split:], nodes_files[split:], edges_files[split:])):
        raw_image = np.array(Image.open(raw_file))
        seg_image = np.array(Image.open(seg_file))
        graph = create_graph(nodes_file, edges_file)

        nodes = np.array([pos for node_id, pos in graph.nodes(data="pos")])
        nodes[:, 2] = 0

        edges = np.array(graph.edges, dtype=np.int32)
        edges = edges[:, [2, 0, 1]]
        edges[:, 0] = 2

        mesh = pyvista.PolyData(nodes)
        mesh.lines = edges.flatten()

        patch_extract(test_path, raw_image, seg_image, mesh)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_data(args)
