import networkx as nx


class GraphModel(object):
    def __init__(self, _image, source_label, _labels=None):
        self.h, self.w = _image.shape
        self.image = _image
        self.labels = _labels
        self.s_label = source_label
        self.G = nx.DiGraph()
        self.init_graph()

    def init_graph(self):
        self.G.add_node("source")
        self.G.add_node("target")

        edges = []
        for i in range(self.h):
            for j in range(self.w):
                neighbours = self.get_neighbours(i, j, self.h, self.w)
                edges += neighbours
                self.G.add_edge("source", i*self.w+j, weight=(self.image[i, j] - self.s_label)**2)
                # todo: if label of node is not alpha, distance to class, if it is alpha, infinty
                self.G.add_edge(i * self.w + j, "target", weight=(self.image[i, j] - self.s_label)**2)
        self.G.add_edges_from(edges)

    def cut_graph(self):
        """Return label of each image node (source/target) using maxflow-mincut"""
        labels = np.zeros((self.h, self.w))
        # if a node is clustered with source, it takes the source label
        # otherwise it keeps its previous label (for alpha-expansion) or take target label (0)
        return labels

    def get_neighbours(self, row, col, height, width):
        pos = row*width + col
        res = []
        if row != 0:
            n = (row-1)*width + col
            res.append((pos, n, {'weight': (self.image[row, col] - self.image[row-1, col])**2}))
        if row != height-1:
            n = (row+1)*width + col
            res.append((pos, n, {'weight': (self.image[row, col] - self.image[row+1, col])**2}))
        if col != 0:
            res.append((pos, pos-1, {'weight': (self.image[row, col] - self.image[row, col-1])**2}))
        if col != width-1:
            res.append((pos, pos+1, {'weight': (self.image[row, col] - self.image[row, col+1])**2}))
        return res

    @staticmethod
    def get_pos(row, col, width):
        return row*width + col

    @staticmethod
    def get_coord(pos, width):
        row = pos // width
        col = pos % width
        return row, col
