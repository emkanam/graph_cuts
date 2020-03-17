import networkx as nx
import numpy as np
import maxflow
from time import time

INFINITY = float('inf')


class GraphModel(object):
    def __init__(self, _image, source_label, _labels=None):
        self.h, self.w = _image.shape
        self.image = _image
        self.labels = _labels
        self.s_label = source_label
        # create graph
        self.G = nx.grid_2d_graph(self.h, self.w, create_using=nx.DiGraph)
        self.G.add_node("s")  # source node
        self.G.add_node("t")  # target node
        self.init_graph()
        self.init_maxflow_weights()

    def init_graph(self):
        nodes = self.G.nodes  # graph nodes
        # create edges
        for node in nodes:
            if node not in ["s", "t"]:
                self.G.add_edge("s", node)  # edge from source to current node
                self.G.add_edge(node, "t")  # edge from current to target node

        # add weights
        for edge in self.G.edges():
            if not ("t" in edge or "s" in edge):
                start, end = edge
                im_xy1 = self.image[start]
                im_xy2 = self.image[end]
                weight = (im_xy1 - im_xy2) ** 2
                self.G[start][end]['weight'] = weight

    def init_weights(self, labels):
        to_remove = []
        self.labels = labels
        for edge in self.G.edges():
            start, end = edge
            # add s-t weights
            if "t" in edge or "s" in edge:
                weight = INFINITY
                if "s" == start:
                    im_xy = self.image[end]
                    weight = (im_xy - self.s_label) ** 2

                if "t" == end:
                    lab_xy = labels[start]
                    if lab_xy != self.s_label:
                        weight = (self.image[start] - lab_xy) ** 2  # D_p(f_p)
                self.G[start][end]['weight'] = weight

    def init_maxflow_weights(self):
        # Edges pointing right
        self.up_structure = np.zeros((3, 3))
        self.down_structure = np.zeros((3, 3))
        self.left_structure = np.zeros((3, 3))
        self.right_structure = np.zeros((3, 3))
        self.up_structure[0, 1] = 1
        self.down_structure[2, 1] = 1
        self.left_structure[1, 0] = 1
        self.right_structure[1, 2] = 1

        self.up_weights = np.zeros(self.image.shape)
        self.down_weights = np.zeros(self.image.shape)
        self.left_weights = np.zeros(self.image.shape)
        self.right_weights = np.zeros(self.image.shape)

        self.up_weights[1:, :] = (self.image[1:, :] - self.image[:-1, :]) ** 2
        self.down_weights[:-1, :] = (self.image[1:, :] - self.image[:-1, :]) ** 2
        self.left_weights[:, 1:] = (self.image[:, 1:] - self.image[:, :-1]) ** 2
        self.right_weights[:, :-1] = (self.image[:, 1:] - self.image[:, :-1]) ** 2

    def get_maxflow_object(self):
        sr_weights = np.zeros((self.h, self.w))
        tr_weights = np.zeros((self.h, self.w))

        for i in range(self.h):
            for j in range(self.w):
                sr_weights[i, j] = self.G['s'][(i, j)]['weight']
                tr_weights[i, j] = self.G[(i, j)]['t']['weight']

        g = maxflow.Graph[int]()
        nodeids = g.add_grid_nodes(self.image.shape)
        g.add_grid_edges(nodeids, structure=self.up_structure, weights=self.up_weights, symmetric=False)
        g.add_grid_edges(nodeids, structure=self.down_structure, weights=self.down_weights, symmetric=False)
        g.add_grid_edges(nodeids, structure=self.left_structure, weights=self.left_weights, symmetric=False)
        g.add_grid_edges(nodeids, structure=self.right_structure, weights=self.right_weights, symmetric=False)
        g.add_grid_tedges(nodeids, sr_weights, tr_weights)

        return g, nodeids

    def cut_graph(self):
        """Return label of each image node (source/target) using maxflow-mincut"""
        start = time()
        labels = np.zeros((self.h, self.w))
        g, nodeids = self.get_maxflow_object()
        g.maxflow()
        sgm = g.get_grid_segments(nodeids)
        for i in range(self.h):
            for j in range(self.w):
                if sgm[i, j]:
                    labels[i, j] = self.s_label
                else:
                    labels[i, j] = self.labels[i, j]
        # if a node is clustered with source, it takes the source label
        # otherwise it keeps its previous label (for alpha-expansion) or take target label (0)
        end = time()
        print("Computed graph cut in %f.3(s)" % (end-start))
        return labels

    def get_neighbours(self, row, col, height, width):
        pos = row*width + col
        res = []
        if row != 0:
            n = (row-1)*width + col
            res.append((pos, n, {'weight': (float(self.image[row, col]) - float(self.image[row-1, col]))**2}))
        if row != height-1:
            n = (row+1)*width + col
            res.append((pos, n, {'weight': (float(self.image[row, col]) - float(self.image[row+1, col]))**2}))
        if col != 0:
            res.append((pos, pos-1, {'weight': (float(self.image[row, col]) - float(self.image[row, col-1]))**2}))
        if col != width-1:
            res.append((pos, pos+1, {'weight': (float(self.image[row, col]) - float(self.image[row, col+1]))**2}))
        return res

    @staticmethod
    def get_pos(row, col, width):
        return row*width + col

    @staticmethod
    def get_coord(pos, width):
        row = pos // width
        col = pos % width
        return row, col
