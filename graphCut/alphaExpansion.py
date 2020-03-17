import numpy as np
from graphModel import GraphModel


def alpha_expansion(_image, max_it = 1000, levels=[0, 51, 102, 153, 255], count_cond=5):
    """Performs multi-label segmentation with alpha expansion algorithm (based on GraphModel class)"""
    labels = np.random.randint(5, size=_image.shape)
    count = 0
    for it in range(max_it):
        # select an alpha class randomly
        alpha_class = levels[np.random.randint(len(levels))]
        # initialize a graph for binary segmentation alpha vs not alpha
        graph = GraphModel(_image, alpha_class, _labels=labels)
        graph.init_graph()
        # perform graph cut, pixels that remain linked with alpha are labeled alpha, others that were previously labeled
        # not alpha, keep their label if not linked with alpha
        new_labels = graph.cut_graph()

        diff = labels - new_labels
        if diff.all() == 0:  # if the labels did not change
            count += 1  # after count_cond=5 update without any change in labels, end the iteration
        else:
            count = 0
        if count > count_cond:
            break
        labels = new_labels
    return labels
