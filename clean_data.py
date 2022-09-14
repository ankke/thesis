import sys

import numpy as np
import yaml
import os
import torch
import imageio
import pyvista
from utils import dict2obj
from argparse import ArgumentParser

def evaluate_single_sample(seg, nodes, strictness=1):
    for node in nodes:
        x = node[0]
        y = node[1]
        if np.amax(seg[0, y - strictness:y + strictness, x - strictness: x + strictness]) == 0:
            return False
    return True

def remove_bad_samples(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)

    img_folder = os.path.join(args.source, 'raw')
    seg_folder = os.path.join(args.source, 'seg')
    vtk_folder = os.path.join(args.source, 'vtp')

    img_files_good = []
    vtk_files_good = []
    seg_files_good = []
    img_files_bad = []
    vtk_files_bad = []
    seg_files_bad = []

    i = 0

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]

        img_file = os.path.join(img_folder, file_+'data.png')
        vtk_file = os.path.join(vtk_folder, file_ + 'graph.vtp')
        seg_file = os.path.join(seg_folder, file_ + 'seg.png')

        image_data = imageio.imread(img_file)
        image_data = np.transpose(np.asarray(image_data, dtype=np.float), (2, 0, 1))

        vtk_data = pyvista.read(vtk_file)
        nodes = np.asarray(vtk_data.points, dtype=np.float)[:, :2]

        # Bring to pixel raster
        # Re-order coordinates such that (0,0) is at top left, stretched to image size, round to int
        tmp = np.copy(nodes[:, 0])
        nodes[:, 0] = np.rint(nodes[:, 1] * image_data.shape[2])
        nodes[:, 1] = np.rint(tmp * image_data.shape[1])
        nodes = nodes.astype(int)
        edges = np.asarray(vtk_data.lines.reshape(-1, 3), dtype=np.int)[:, 1:]

        seg_data = imageio.imread(seg_file)
        seg_data = np.expand_dims(np.asarray(seg_data, dtype=np.int), 0)

        print(f"node: {i}")
        print(nodes)
        print(edges)

        if evaluate_single_sample(seg_data, nodes):
            img_files_good.append(os.path.join(img_folder, file_ + 'data.png'))
            vtk_files_good.append(os.path.join(vtk_folder, file_ + 'graph.vtp'))
            seg_files_good.append(os.path.join(seg_folder, file_ + 'seg.png'))
        else:
            img_files_bad.append(os.path.join(img_folder, file_ + 'data.png'))
            vtk_files_bad.append(os.path.join(vtk_folder, file_ + 'graph.vtp'))
            seg_files_bad.append(os.path.join(seg_folder, file_ + 'seg.png'))

        #np.set_printoptions(threshold=sys.maxsize)

        if i >= 2:
            break
        else:
            i += 1

    print(len(img_files_good))
    print(img_files_bad)

    """data_dicts = [
        {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in
        zip(img_files_good, vtk_files_good, seg_files_good)
    ]"""


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

if __name__ == "__main__":
    args = parser.parse_args()
    remove_bad_samples(args)
