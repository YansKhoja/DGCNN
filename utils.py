from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
import networkx as nx
import pydot

class TransFormGetDegree(object):

    def __call__(self, data):
        col, nodes, x = data.edge_index[1], data.num_nodes, data.x
        deg = degree(col, nodes)
        deg = deg / deg.max()
        if data.x is None:
            data.x = deg.view(-1, 1)
        return data

# Fonction de visualisation de graph - inutilisé
"""
def visualize(h, color, labels=None, wlabels=False, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), labels=None, with_labels=wlabels, cmap="Set2")
    plt.show()
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G, pos=nx.nx_pydot.pydot_layout(G, prog="dot"), labels=None, with_labels=True)
    visualize(G, color=data.y,wlabels=True)
"""