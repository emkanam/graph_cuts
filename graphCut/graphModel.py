import networkx as nx


class GraphModel(object):
    def __init__(self, _image):
        self.h, self.w = _image.shape
        self.image = _image
        self.G = nx.DiGraph()
        self.init_graph()

    def init_graph(self):
        self.G.add_node("source")
        self.G.add_node("target")

        edges = []
        for i in range(self.h):
            for j in range(self.w):
                neighbours = GraphModel.get_neighbours(i, j, self.h, self.w)
                edges += neighbours
                self.G.add_edge("source", i*self.w+j, weight=0)
                self.G.add_edge(i * self.w + j, "target", weight=1)
        self.G.add_edges_from(edges)

    @staticmethod
    def get_pos(row, col, width):
        return row*width + col

    @staticmethod
    def get_coord(pos, width):
        row = pos // width
        col = pos % width
        return row, col

    # @staticmethod
    def get_neighbours(row, col, height, width):
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

