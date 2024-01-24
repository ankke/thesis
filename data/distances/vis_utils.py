import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx

from torch_geometric.utils import to_networkx
from tqdm import tqdm

from data.visualize_sample import draw_graph


def get_features_df(results):
    df = pd.DataFrame(results)

    df['num_nodes_graph1'] = df['pair'].apply(lambda pair: pair[0].num_nodes)
    df['num_edges_graph1'] = df['pair'].apply(lambda pair: pair[0].num_edges)
    df['conn_comp_graph1'] = df['pair'].apply(lambda pair: len(list(nx.connected_components(to_networkx(pair[0]).to_undirected()))))
#     df['diameter_graph1'] = df['pair'].apply(lambda pair: len(list(nx.diameter(to_networkx(pair[0]).to_undirected()))))
#     df['avg_s_path_graph1'] = df['pair'].apply(lambda pair: len(list(nx.average_shortest_path_length(to_networkx(pair[0]).to_undirected()))))

    df['num_nodes_graph2'] = df['pair'].apply(lambda pair: pair[1].num_nodes)
    df['num_edges_graph2'] = df['pair'].apply(lambda pair: pair[1].num_edges)
    df['conn_comp_graph2'] = df['pair'].apply(lambda pair: len(list(nx.connected_components(to_networkx(pair[1]).to_undirected()))))
#     df['diameter_graph2'] = df['pair'].apply(lambda pair: len(list(nx.diameter(to_networkx(pair[1]).to_undirected()))))
#     df['avg_s_path_graph2'] = df['pair'].apply(lambda pair: len(list(nx.average_shortest_path_length(to_networkx(pair[1]).to_undirected()))))

    return df


def plot_ged_res(results):
    plot_res(results, method='ged')

def plot_res(results, method):
    rows = len(results)
    
    fig = plt.figure()
    fig.set_size_inches(8, 4 * rows)

    subfigs = fig.subfigures(rows, 1)

    for i, res in enumerate(tqdm(results)):
        g1 = res['pair'][0]
        g2 = res['pair'][1]
        distance = res[method]
        
        (ax1, ax2) = subfigs[i].subplots(1, 2)

        draw_graph(g1.pos, g1.edge_index.t(), ax1)
        ax1.title.set_text(f"N: {res['num_nodes_graph1']}, E: {res['num_edges_graph1']}, C: {res['conn_comp_graph1']}")

        draw_graph(g2.pos, g2.edge_index.t(), ax2)
        ax2.title.set_text(f"N: {res['num_nodes_graph2']}, E: {res['num_edges_graph2']}, C: {res['conn_comp_graph2']}")
        
        subfigs[i].suptitle(f"{method}: {distance}", fontsize=12)


def plot_corr_heatmat(df, method='ged'):
    columns_of_interest = [method, 'num_nodes_graph2', 'num_edges_graph2', 'conn_comp_graph2']

    correlation_matrix = df[columns_of_interest].corr()
    ged_correlations = correlation_matrix[method].drop(method)
    
    plt.figure(figsize=(1, 5))
    sns.heatmap(ged_correlations.to_frame(), annot=True, cmap='coolwarm', fmt=".2f", cbar=False)
    plt.show()


def plot_components(g, ax):
    g_nx = to_networkx(g).to_undirected()
    connected_components = list(nx.connected_components(g_nx))
    
    for component_nodes in connected_components:
        
        subgraph = g_nx.subgraph(component_nodes)
        component_nodes_srt = sorted(list(component_nodes))
        
        translated_edges = [(component_nodes_srt.index(e_1), component_nodes_srt.index(e_2)) for e_1, e_2 in subgraph.edges]
        translated_pos = g.pos[component_nodes_srt]
        
        draw_graph(translated_pos, translated_edges, ax)

def plot_with_conn_comp(results, method='ged'):
    rows = len(results)
    
    fig = plt.figure()
    fig.set_size_inches(8, 4 * rows)

    subfigs = fig.subfigures(rows, 1)

    for i, res in enumerate(tqdm(results)):
        g1 = res['pair'][0]
        g2 = res['pair'][1]
        distance = res[method]
        
#         ((ax1, ax2), (ax3, ax4)) = subfigs[i].subplots(2, 2)
        ax1, ax2 = subfigs[i].subplots(1, 2)
        
        plot_components(g1, ax1)
        plot_components(g2, ax2)
        
#         draw_graph(g1.pos, g1.edge_index.t(), ax3)
#         draw_graph(g2.pos, g2.edge_index.t(), ax4)
        
        subfigs[i].suptitle(f"{method}: {distance}, conn comp 1: {res['conn_comp_graph1']}, conn comp 2: {res['conn_comp_graph2']}", fontsize=12)