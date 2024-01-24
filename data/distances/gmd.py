# adapted from https://github.com/sushovan4/ggdlib/blob/main/src/GMD.py

import os
import torch
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from time import time

from torch_geometric.utils import to_networkx
from multiprocessing import Pool
import networkx as nx
import torch

from time import time
from multiprocessing import Pool, cpu_count
from torch_geometric.utils import to_networkx, from_networkx


NUM_PROCESSES = cpu_count()

def distance(G1, idx1, G2, idx2):
    pos1 = nx.get_node_attributes(G1,'pos')
    pos2 = nx.get_node_attributes(G2,'pos')
    return np.linalg.norm(np.array(pos1[idx1]) - np.array(pos2[idx2]))

def pdd(G):
    adj = nx.adjacency_matrix(G)
    num_nodes = G.number_of_nodes()
    res = np.zeros((num_nodes, num_nodes - 1))
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = distance(G, i, G, j)
            if(j < i):
                res[i, j] = adj[i, j] * dist
            if(j > i):
                res[i, j - 1] = adj[i, j] * dist
    return res

def _compute_gmd(args):
    try:
        data_pair, C_V, C_E, M = args
        G1 = to_networkx(data_pair[0], node_attrs=["pos"]).to_undirected()
        G2 = to_networkx(data_pair[1], node_attrs=["pos"]).to_undirected()

        num_nodes1, num_nodes2 = G1.number_of_nodes(), G2.number_of_nodes()
        p = min(num_nodes1, num_nodes2)
        pdd1, pdd2 = pdd(G1), pdd(G2)    
        G = nx.DiGraph()

        for node in G1.nodes:
            G.add_node('g1.' + str(node), demand = -1 * M)

        G.add_node("eps1", demand = -num_nodes2 * M)

        for node in G2.nodes:
            G.add_node('g2.' + str(node), demand = 1 * M)
        
        G.add_node("eps2", demand = num_nodes1 * M)

        for i in range(num_nodes1):
            for j in range(num_nodes2):
                weight = C_V * distance( G1, i, G2, j)
                weight += C_E * 0.5 * np.linalg.norm( pdd1[i, :] @ np.eye(num_nodes1-1, p) - pdd2[j, :] @ np.eye(num_nodes2-1, p), ord = 1)

                G.add_edge('g1.' + str(i), 'g2.' + str(j), weight = round( weight * M) )

        for j in range(num_nodes2):
            weight =  C_E * np.linalg.norm( pdd2[j, :], ord = 1 )
            G.add_edge("eps1", 'g2.' + str(j), weight = round( weight * M))

        for i in range(num_nodes1):
            weight = C_E * np.linalg.norm( pdd1[i, :], ord = 1 )
            G.add_edge('g1.' + str(i), "eps2", weight = round( weight * M))
        
        G.add_edge("eps1", "eps2", weight = 0)

        cost, flow = nx.network_simplex(G)
        cost  = cost / (M * M)

        return {'pair': data_pair, 'gmd': cost, 'gmd_flow': flow}
    except:
        pass


def _gmd(graph_pairs, C_V, C_E, M):
    args = [(graph_pair, C_V, C_E, M) for graph_pair in graph_pairs]
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(_compute_gmd, args, chunksize=1)
    return results


def generate_gmd(graph_pairs, dataset_name, dir, meta_data, C_V, C_E, M):
    num_pairs = len(graph_pairs)

    file_name = f'{dataset_name}_n{num_pairs}_CV{C_V}_CE{C_E}_M{M}.pth'
    file_path = os.path.join(dir, file_name)

    start_time = time()

    ged_res = _gmd(graph_pairs, C_V=C_V, C_E=C_E, M=M)

    end_time = time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.1f} seconds")

    meta_data['C_V'] = C_V
    meta_data['C_E'] = C_V
    meta_data['M'] = M

    torch.save((ged_res, meta_data), file_path)

    return file_path

def random_pos_shift(graph, radius):
    for node, attributes in graph.nodes(data=True):
        if 'pos' in attributes:
            current_position = np.array(attributes['pos'])
            shift = np.random.uniform(-radius, radius, size=(2,))
            new_position = current_position + shift
            graph.nodes[node]['pos'] = new_position

def e2e_gmd(dataset, id_1, id_2, remove=0, random_radius=None, C_V=1.0, C_E=1.0, multiplier=1.0, shift=1):
    G1 = to_networkx(dataset[id_1], node_attrs=["pos"]).to_undirected()
    G2 = to_networkx(dataset[id_2], node_attrs=["pos"]).to_undirected()
    
    if random_radius is not None:
        random_pos_shift(G2, random_radius)
    
    nn = G2.number_of_nodes()

    for i in range(1, remove + 1):
        G2.remove_node(nn - i)
    
    start_time = time()
    res = _compute_gmd(((from_networkx(G1), from_networkx(G2)), C_V, C_E, multiplier))
    end_time = time()
    execution_time = end_time - start_time
    print(f"GMD execution time: {execution_time:.2f} seconds")
    
    ged = nx.graph_edit_distance(G1, G2, timeout=10)
    
    print(f"GMD: {res['gmd']}, GED: {ged}")
    vis_mapping(G1, G2, res['gmd_flow'], shift)


def vis_mapping(G1, G2, flow, shift, ax=None):
    mapping = {node: 'g1.' + str(node) for node in G1.nodes}
    G1 = nx.relabel_nodes(G1, mapping)
    mapping = {node: 'g2.' + str(node) for node in G2.nodes}
    G2 = nx.relabel_nodes(G2, mapping)
    
    for u in flow.values():
        for v, w in u.items():
            u[v] = { 'weight': w } 
    F = nx.DiGraph(flow)
    F.remove_node("eps1")
    F.remove_node("eps2")
    F.remove_edges_from([(n1, n2) for n1, n2, w in F.edges(data="weight") if w == 0])
    pos1 = nx.get_node_attributes(G1,'pos')
    pos2 = nx.get_node_attributes(G2,'pos')

    for k in pos2.keys():
            pos2[k] = np.add( pos2[k], (shift,0))
    pos = pos1 | pos2
    
    if ax is not None:
        plt.figure(figsize=(6, 4))

    nx.draw(G1, pos1, edge_color = "red", node_color = "red", node_size = 20, ax=ax)
    nx.draw(G2, pos2, edge_color = "blue", node_color = "blue", node_size = 20, ax=ax)
    nx.draw(F, pos, edge_color = "gray", width = 1, style = '--', node_size = 0, alpha = 0.5, connectionstyle="arc3, rad=0.2", arrowsize = 17, ax=ax)


def vis_res_mapping(results, shift=1):
    rows = len(results)
    
    fig = plt.figure()
    fig.set_size_inches(8, 4 * rows)

    subfigs = fig.subfigures(rows, 1)

    for i, res in enumerate(results):
        g1 = res['pair'][0]
        g2 = res['pair'][1]

        G1 = to_networkx(g1, node_attrs=["pos"]).to_undirected()
        G2 = to_networkx(g2, node_attrs=["pos"]).to_undirected()

        distance = res['gmd']
        
        ax = subfigs[i].subplots(1, 1)

        vis_mapping(G1, G2, res['gmd_flow'], shift, ax)
        # ax.title.set_text(f"1 N: {res['num_nodes_graph1']}, E: {res['num_edges_graph1']}, C: {res['conn_comp_graph1']} \n 2 N: {res['num_nodes_graph2']}, E: {res['num_edges_graph2']}, C: {res['conn_comp_graph2']}")

        subfigs[i].suptitle(f"GMD: {distance} \n Left: N: {res['num_nodes_graph1']}, E: {res['num_edges_graph1']}, C: {res['conn_comp_graph1']} \n Right: N: {res['num_nodes_graph2']}, E: {res['num_edges_graph2']}, C: {res['conn_comp_graph2']}", fontsize=12)