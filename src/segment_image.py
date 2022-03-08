from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from src.utils.maincolors import get_main_colors, compute_masks, compute_distributions
from src.utils.graphutils import build_base_graph, add_data_edges
from src.utils.aexpansion import (
    make_arbitrary_partition,
    make_expansion,
    show_segmentation,
)


def read_image(frame: str) -> np.ndarray:
    frame = frame.zfill(6)
    return plt.imread("./data/VOCdevkit/VOC2007/JPEGImages/" + frame + ".jpg")


def run_expansion(
    base_graph: nx.Graph,
    distributions: List[Dict[str, np.ndarray]],
    alpha: int,
    image: np.ndarray,
    original_parition: Dict[int, set],
    params: dict,
) -> Tuple[Dict[int, set], nx.Graph]:
    """Runs a full alpha-expansion iteration

    Args:
        base_graph (nx.Graph): Base graph
        distributions (List[Dict[str, np.ndarray]]): Main colors distributions
        alpha (int): Alpha iteration label
        image (np.ndarray): Image
        original_parition (Dict[int, set]): Starting partition
        params (dict): Params

    Returns:
        Tuple[Dict[int, set], nx.Graph]: New partition and new graph
    """
    new_graph, auxiliary_nodes = add_data_edges(
        base_graph, distributions, alpha, image, original_parition, params
    )
    _, binary_partition = nx.algorithms.minimum_cut(new_graph, "-1", "-2")
    final_partition = make_expansion(
        original_parition, binary_partition, alpha, auxiliary_nodes
    )
    return final_partition, new_graph


if __name__ == "__main__":
    dummy_image = read_image("245")

    params = {"n_colors": 5, "bins": 50, "epsilon": 0.3, "lambda": 10}

    print("Computing main colors...")
    reshaped_image = dummy_image.reshape((-1, 3))
    main_colors = get_main_colors(reshaped_image, params["n_colors"], params["bins"])

    masks = compute_masks(main_colors, dummy_image)
    distributions = compute_distributions(masks, dummy_image)

    print("Building base graph...")
    base_graph = build_base_graph(dummy_image)

    original_partition = make_arbitrary_partition(dummy_image, main_colors)

    show_segmentation(dummy_image, original_partition)

    partitions = [original_partition]
    energies = []
    for k in range(1):
        for alpha in range(params["n_colors"]):
            print("Alpha:", alpha)
            final_partition, new_graph = run_expansion(
                base_graph, distributions, alpha, dummy_image, partitions[-1], params
            )
            partitions.append(final_partition)
            break

    show_segmentation(dummy_image, final_partition)
    # edge_labels = nx.get_edge_attributes(new_graph,'capacity')
    # pos = nx.spring_layout(new_graph)
    # nx.draw(new_graph, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(new_graph, pos, edge_labels)
    # plt.draw()
    plt.show()
