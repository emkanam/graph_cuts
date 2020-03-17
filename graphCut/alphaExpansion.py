import numpy as np
from graphModel import GraphModel


def alpha_expansion(_image, max_it=1000, levels=[0, 51, 102, 153, 255], count_cond=5):
    """Performs multi-label segmentation with alpha expansion algorithm (based on GraphModel class)"""
    n_levels = len(levels)  # total levels
    labels = np.random.choice(levels, size=_image.shape)  # each pixel gets random level
    graph = GraphModel(_image, 0, _labels=labels)
    count = 0
    for it in range(max_it):
        # select an alpha class randomly
        alpha_class = levels[np.random.randint(n_levels)]
        # initialize a graph for binary segmentation alpha vs not alpha
        graph.s_label = alpha_class
        graph.init_weights(labels)
        # perform graph cut, pixels that remain linked with alpha are labeled alpha, others that were previously labeled
        # not alpha, keep their label if not linked with alpha
        new_labels = graph.cut_graph()
        
        if np.array_equal(labels, new_labels):  # if the labels did not change
            count += 1  # after count_cond=5 update without any change in labels, end the iteration
        else:
            count = 0
        if count > count_cond:
            break
        labels = new_labels
    return labels
