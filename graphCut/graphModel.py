import networkx as nx
import numpy as np


class GraphModel(object):
    def __init__(self, _image):
        self.h, self.w = _image.shape
        self.image = _image
        self.s_val = 0.0
        self.t_val = 1.0
        self.G = nx.grid_2d_graph(self.h, self.w, create_using=nx.DiGraph)
        self.G.add_node("s")  # source node
        self.G.add_node("t")  # target node
        self.init_graph()

    def init_graph(self):
        nodes = self.G.nodes  # graph nodes
        # create edges
        for node in nodes:
            if node not in ["s", "t"]:
                # compute affinity
                im_xy = self.image[node[0], node[1]]
                s_weight = np.exp(-(self.s_val-im_xy)**2)
                t_weight = np.exp(-(self.t_val-im_xy)**2)
                # add edge and weight
                self.G.add_edge("s", node, weight=s_weight)  # edge from source to current node
                self.G.add_edge(node, "t", weight=t_weight)  # edge from current to target node

        # add weights
        for edge in self.G.edges():
            if not ("t" in edge or "s" in edge):
                coord1, coord2 = edge
                im_xy1 = self.image[coord1[0], coord1[1]]
                im_xy2 = self.image[coord2[0], coord2[1]]
                weight = np.exp(-(im_xy1-im_xy2)**2)
                self.G[coord1][coord2]['weight'] = weight
    def cut_graph(self):
        """Return label of each image node (source/target) using maxflow-mincut"""
        labels = np.zeros((self.h, self.w))
        # if a node is clustered with source, it takes the source label
        # otherwise it keeps its previous label (for alpha-expansion) or take target label (0)
        return labels

    @staticmethod
    def get_coord(pos, width):
        row = pos // width
        col = pos % width
        return row, col

    @staticmethod
    def get_neighbours(row, col, height, width):
        pos = row*width + col
        res = []
        if row != 0:
            n = (row-1)*width + col
            res.append((pos, n))
        if row != height-1:
            n = (row+1)*width + col
            res.append((pos, n))
        if col != 0:
            res.append((pos, pos-1))
        if col != width-1:
            res.append((pos, pos+1))
        return res