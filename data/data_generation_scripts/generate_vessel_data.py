import math
import imageio
import pyvista
import numpy as np
import pickle
import random
import os
from argparse import ArgumentParser

patch_size = [128, 128, 1]
pad = [5, 5, 0]


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, a_min=-1, a_max=1))

vector_norm = 25.0


def neighbor_transpos(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (k[1], k[0])
        nv = []

        for _v in v:
            nv.append((_v[1], _v[0]))

        n_out[nk] = nv

    return n_out


def neighbor_to_integer(n_in):
    n_out = {}

    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))

        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []

        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))

            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)

        n_out[nk] = nv

    return n_out


def save_input(path, idx, patch, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    imageio.imwrite(path+'raw/sample_'+str(idx).zfill(6)+'_data.png', patch)

    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)

    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'mesh/sample_'+str(idx).zfill(4)+'_segmentation.stl')

    patch_edge = np.concatenate(
        (np.int32(2*np.ones((patch_edge.shape[0], 1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image,  mesh, device=None):
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
    x_ = np.int32(np.linspace(5, h-5-p_h, 32))
    y_ = np.int32(np.linspace(5, w-5-p_w, 32))

    ind = np.meshgrid(x_, y_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(2, -1).T)):
        start = np.array((start[0], start[1], 0))
        end = start + np.array(patch_size)-1 - 2*np.array(pad)

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1] +
                       p_w, ], ((pad_h, pad_h), (pad_w, pad_w), ))
        patch_list = [patch]
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

        mod_patch_coord_list, mod_patch_edge_list = prune_patch(
            patch_coord_list, patch_edge_list)

        # save data
        for patch, patch_coord, patch_edge in zip(patch_list, mod_patch_coord_list, mod_patch_edge_list):
            if patch.sum() > 10:
                save_input(save_path, image_id, patch,
                           patch_coord, patch_edge)
                image_id = image_id+1
                # print('Image No', image_id)


def prune_patch(patch_coord_list, patch_edge_list):
    """[summary]

    Args:
        patch_list ([type]): [description]
        patch_coord_list ([type]): [description]
        patch_edge_list ([type]): [description]

    Returns:
        [type]: [description]
    """
    mod_patch_coord_list = []
    mod_patch_edge_list = []

    for coord, edge in zip(patch_coord_list, patch_edge_list):

        # find largest graph segment in graph and in skeleton and see if they match
        dist_adj = np.zeros((coord.shape[0], coord.shape[0]))
        dist_adj[edge[:, 0], edge[:, 1]] = np.sum(
            (coord[edge[:, 0], :]-coord[edge[:, 1], :])**2, 1)
        dist_adj[edge[:, 1], edge[:, 0]] = np.sum(
            (coord[edge[:, 0], :]-coord[edge[:, 1], :])**2, 1)

        # straighten the graph by removing redundant nodes
        start = True
        node_mask = np.ones(coord.shape[0], dtype=bool)
        while start:
            degree = (dist_adj > 0).sum(1)
            deg_2 = list(np.where(degree == 2)[0])
            if len(deg_2) == 0:
                start = False
            for n, idx in enumerate(deg_2):
                deg_2_neighbor = np.where(dist_adj[idx, :] > 0)[0]

                p1 = coord[idx, :]
                p2 = coord[deg_2_neighbor[0], :]
                p3 = coord[deg_2_neighbor[1], :]
                l1 = p2-p1
                l2 = p3-p1
                node_angle = angle(l1, l2)*180 / math.pi
                if node_angle > 160:
                    node_mask[idx] = False
                    dist_adj[deg_2_neighbor[0],
                             deg_2_neighbor[1]] = np.sum((p2-p3)**2)
                    dist_adj[deg_2_neighbor[1],
                             deg_2_neighbor[0]] = np.sum((p2-p3)**2)

                    dist_adj[idx, deg_2_neighbor[0]] = 0.0
                    dist_adj[deg_2_neighbor[0], idx] = 0.0
                    dist_adj[idx, deg_2_neighbor[1]] = 0.0
                    dist_adj[deg_2_neighbor[1], idx] = 0.0
                    break
                elif n == len(deg_2)-1:
                    start = False

        new_coord = coord[node_mask, :]
        new_dist_adj = dist_adj[np.ix_(node_mask, node_mask)]
        new_edge = np.array(np.where(np.triu(new_dist_adj) > 0)).T

        mod_patch_coord_list.append(new_coord)
        mod_patch_edge_list.append(new_edge)

    return mod_patch_coord_list, mod_patch_edge_list


parser = ArgumentParser()
parser.add_argument('--source',
                    default=None,
                    required=True,
                    help='Path to source directory')
parser.add_argument('--target',
                    default=None,
                    required=True,
                    help='Path to target directory')
parser.add_argument('--source_number',
                    default=None,
                    type=int,
                    required=True,
                    help='Number of images in source directory')
parser.add_argument('--seed',
                    default=0,
                    type=int,
                    required=False,
                    help='Random seed')
parser.add_argument('--split',
                    default=0.8,
                    type=float,
                    help='Train/Test split. 0.8 means 80% of the data will be training data and 20% testing data'
                        'Default: 0.8')

image_id = 1

def generate_data(args):
    global image_id

    root_dir = args.source
    target_dir = args.target
    amount_images = args.source_number
    split = args.split

    # Sets the seed for reproducibility
    random.seed(args.seed)

    indrange_train = []
    indrange_test = []

    for x in range(amount_images):
        if random.random() <= split:
            indrange_train.append(x)
        else:
            indrange_test.append(x)

    image_id = 1
    train_path = f"{target_dir}/train_data/"
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path + '/raw')
        os.makedirs(train_path + '/vtp')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')

    raw_files = []
    vtk_files = []

    for ind in indrange_train:
        raw_files.append(f"{root_dir}/{ind}_scan")
        vtk_files.append(f"{root_dir}/{ind}_gt_graph.pkl")

    for ind in range(len(raw_files)):
        print(ind)
        try:
            sat_img = imageio.imread(raw_files[ind] + ".png")
        except:
            sat_img = imageio.imread(raw_files[ind] + ".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)
        node_array = graph["nodes"] * np.array([sat_img.shape[0], sat_img.shape[1]])
        edge_array = graph["edges"]

        patch_coord = np.concatenate(
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()
        patch_extract(train_path, sat_img, mesh)

    image_id = 1
    test_path = f"{target_dir}/test_data/"
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path + '/vtp')
        os.makedirs(test_path + '/raw')
    else:
        raise Exception("Test folder is non-empty")

    print('Preparing Test Data')

    raw_files = []
    vtk_files = []

    for ind in indrange_test:
        # If data is prefixed with city name, add this to the filenames
        raw_files.append(f"{root_dir}/{ind}_scan")
        vtk_files.append(f"{root_dir}/{ind}_gt_graph.pkl")

    for ind in range(len(raw_files)):
        print(ind)
        try:
            sat_img = imageio.imread(raw_files[ind] + ".png")
        except:
            sat_img = imageio.imread(raw_files[ind] + ".jpg")

        with open(vtk_files[ind], 'rb') as f:
            graph = pickle.load(f)
        node_array = graph["nodes"] * np.array([sat_img.shape[0], sat_img.shape[1]])
        edge_array = graph["edges"]

        patch_coord = np.concatenate(
            (node_array, np.int32(np.zeros((node_array.shape[0], 1)))), 1)
        mesh = pyvista.PolyData(patch_coord)
        patch_edge = np.concatenate(
            (np.int32(2 * np.ones((edge_array.shape[0], 1))), edge_array), 1)
        mesh.lines = patch_edge.flatten()
        patch_extract(test_path, sat_img, mesh)


if __name__ == "__main__":
    args = parser.parse_args()
    generate_data(args)
