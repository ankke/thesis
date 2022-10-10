from typing import Generator

import os
import glob
import pickle
import sys
import logging
import numpy as np

from .arterial_tree import Node, ArterialTree

sys.setrecursionlimit(10000)


class Forest:

    def __init__(self, arterial=True, init_file=None):

        self.trees: list[ArterialTree] = []
        self.sim_space = None
        self.simspace_size = None
        self.sim_scale = None
        self.size = None
        self.size_x, self.size_y, self.size_z = tuple((None,None,None))
        self.arterial = arterial

        if init_file:
            self._initialize_from_file(config={'filepath': init_file})


    def _initialize_from_file(self, config):
        print(os.path.abspath(config['filepath']))
        tree_files = glob.glob(os.path.join(config['filepath'], '*.pkl'))
        print(tree_files)

        for tree_file in tree_files:

            f = open(tree_file, 'rb')
            tree = pickle.load(f)
            f.close()

            self.trees.append(tree)


    def get_trees(self):
        return self.trees

    def get_nodes(self) -> Generator[Node, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node

    def get_node_coords(self) -> Generator[np.ndarray, None, None]:
        for tree in self.trees:
            for node in tree.get_tree_iterator(exclude_root=False, only_active=False):
                yield node.position

    def save(self, save_directory='.'):
        name = f'{"Arterial" if self.arterial else "Venous"}Forest'
        os.makedirs(save_directory, exist_ok=True)
        filepath = os.path.join(save_directory, name + '.pkl')
        tmp = self.sim_space
        self.sim_space = None
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        self.simspace = tmp
        logging.info('Saved {} to {}.'.format(name, os.path.abspath(filepath)))
