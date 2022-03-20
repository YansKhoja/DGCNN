from torch_geometric.utils import degree

class TransFormGetDegree(object):

    def __call__(self, data):
        col, nodes, x = data.edge_index[1], data.num_nodes, data.x
        deg = degree(col, nodes)
        deg = deg / deg.max()
        if data.x is None:
            data.x = deg.view(-1, 1)
        return data

