import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, Normalize



def draw_graph(nodes, edges, ax):
    # Create Graph for plotting graphs for actual data and prediction
    cur_graph = nx.Graph()

    # Add all nodes
    for ind, pos in enumerate(nodes):
        # The positions have to be altered so that the graph has the right orientation
        # (all coordinates are normalized to [0,1])
        cur_graph.add_node(ind, pos=(pos[1].item(), 1 - pos[0].item()))

    # Add positions to nodes
    pos = nx.get_node_attributes(cur_graph,'pos')

    # Add all edges
    for edge in edges:
        cur_graph.add_edge(edge[0].item(), edge[1].item())

    # Draw Graphs with matplotlib
    nx.draw_networkx(cur_graph, pos, ax, node_size=20, with_labels=False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, grid_color='r')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def create_sample_visual(samples, number_samples=3):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(3, 3, figsize=(1000 * px, number_samples * 300 * px))

    for i in range(3):
        axs[i, 0].imshow(inv_norm(samples["images"][i].clone().cpu().detach()).permute(1, 2, 0))

        plt.sca(axs[i, 1])
        draw_graph(samples["nodes"][i].clone().cpu().detach(), samples["edges"][i].clone().cpu().detach(), axs[i, 1])
        plt.sca(axs[i, 2])
        draw_graph(samples["pred_nodes"][i], samples["pred_edges"][i], axs[i, 2])

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))


inv_norm = Compose([
    Normalize(
        mean=[0., 0., 0.],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1., 1., 1.]
    ),
])


