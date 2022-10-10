"""
Transforms synthetic vessel data from tree structure, to list and node structure used for the relationformer
"""

import os
import sys
import pickle
import numpy as np
import shutil
from argparse import ArgumentParser

from forest import arterial_tree
from forest import forest

sys.modules['arterial_tree'] = arterial_tree
sys.modules['forest'] = forest

parser = ArgumentParser()
parser.add_argument('--source',
                    default=None,
                    required=True,
                    help='Path to source directory')
parser.add_argument('--target',
                    default=None,
                    required=True,
                    help='Path to target directory')
parser.add_argument('--max_samples',
                    default=None,
                    type=int,
                    required=False,
                    help='Max number of samples that should be prepared')


def convert_subgraph(root, node_number, nodes, edges):
    if len(root.children) == 0:
        return

    cur_len = len(nodes)

    for i, child in enumerate(root.children):
        (x, y, z) = child.position
        nodes.append((x, y))
        edges.append((node_number, cur_len + i))

    for i, child in enumerate(root.children):
        convert_subgraph(child, cur_len + i, nodes, edges)


def transform_vessel_data(args):
    source_counter = 0
    for file in os.scandir(args.source):
        if args.max_samples and source_counter >= args.max_samples:
            break

        if not file.is_dir():
            continue

        nodes = []
        edges = []

        venous_forest = pickle.load(open(f'{file.path}/VenousForest.pkl', 'rb'))
        arterial_forest = pickle.load(open(f'{file.path}/ArterialForest.pkl', 'rb'))

        for forest in [arterial_forest, venous_forest]:
            for tree in forest.get_trees():
                cur = len(nodes)
                (x, y, z) = tree.root.position
                nodes.append((x, y))
                convert_subgraph(tree.root, cur, nodes, edges)

        node_arr = np.array(nodes)
        edge_arr = np.array(edges)

        graph_file = open(f'{args.target}/{source_counter}_gt_graph.pkl', 'wb')

        pickle.dump({
            "nodes": node_arr,
            "edges": edge_arr
        }, graph_file)

        try:
            shutil.copyfile(f'{file.path}/art_ven_gray_z.png', f'{args.target}/{source_counter}_scan.png')
        finally:
            source_counter += 1


if __name__ == "__main__":
    args = parser.parse_args()
    transform_vessel_data(args)

