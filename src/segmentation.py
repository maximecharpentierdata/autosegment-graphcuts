from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import cv2

from src.aexpansion import make_expansion, show_segmentation, construct_segmentation
from src.base_segmentation import kmeans
from src.graph import add_data_edges, build_base_graph, compute_energy
from src.utils import read_image, resize_image

from src.base_segmentation import maincolors


def minimum_cut_networkx(graph: nx.Graph, source: str, target: str):
    _, partition = nx.algorithms.minimum_cut(graph, source, target)
    return partition


def run_expansion(
    base_graph: nx.Graph,
    distributions: List[Dict[str, np.ndarray]],
    alpha: int,
    image: np.ndarray,
    original_parition: Dict[int, set],
    params: dict,
    verbose: bool = True,
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
    if verbose:
        print("Build step graph...")
    new_graph, auxiliary_nodes = add_data_edges(
        base_graph, distributions, alpha, image, original_parition, params, verbose
    )
    if verbose:
        print("Solving minimum cut algorithm...")
    binary_partition = minimum_cut_networkx(new_graph, "-1", "-2")

    final_partition = make_expansion(
        original_parition, binary_partition, alpha, auxiliary_nodes
    )
    return final_partition, new_graph


def segment_image(
    image: np.ndarray, params: dict, verbose: float = True, resize: float = True
) -> dict:
    """Run full segmentation on an image

    Args:
        image (np.ndarray): Input image
        params (dict): Params

    Returns:
        dict: energies contains all energies computed during the segmentation
              partition contains the final partition
    """

    resized_image = resize_image(image, resize)

    if params["method"] == "maincolors":
        distributions, original_partition = maincolors.segmentation(
            resized_image, params, verbose
        )
    elif params["method"] == "kmeans":
        distributions, original_partition = kmeans.segmentation(
            resized_image, params, verbose
        )

    if verbose:
        print("Building base graph...")
    base_graph = build_base_graph(resized_image, verbose)

    if verbose:
        print("Compute original energy...")
    energy = compute_energy(
        distributions, resized_image, original_partition, params, verbose
    )

    if verbose:
        print(
            f"Original data cost: {energy['data_cost']}, Original smooth code: {energy['smooth_cost']}"
        )

    partitions = [original_partition]
    energies = [energy]
    alpha = 0
    for alpha in range(len(distributions)):
        if verbose:
            print("Alpha:", alpha)
        final_partition, new_graph = run_expansion(
            base_graph,
            distributions,
            alpha,
            resized_image,
            partitions[-1],
            params,
            verbose,
        )
        if verbose:
            print("Computing new energy...")
        energy = compute_energy(
            distributions, resized_image, final_partition, params, verbose
        )
        if verbose:
            print(
                f"Data cost: {energy['data_cost']}, Smooth code: {energy['smooth_cost']}"
            )

        if not (energy["data_cost"] + energy["smooth_cost"]) >= (
            energies[-1]["data_cost"] + energies[-1]["smooth_cost"]
        ):
            energies.append(energy)
            partitions.append(final_partition)

    segmented_image = construct_segmentation(partitions[-1], resized_image.shape[:2])
    original_segmented_image = construct_segmentation(
        partitions[0], resized_image.shape[:2]
    )

    if resize:
        segmented_image = cv2.resize(segmented_image, image.shape[:2][::-1]).astype(int)
        original_segmented_image = cv2.resize(
            original_segmented_image, image.shape[:2][::-1]
        ).astype(int)

    return dict(
        energies=energies,
        segmented_image=segmented_image,
        original_segmented_image=original_segmented_image,
    )


def segment_image_star(args):
    return segment_image(*args)


if __name__ == "__main__":
    dummy_image = read_image("250")

    params = {
        "method": "kmeans",
        "bins": 5,
        "n_clusters": 3,
        "lambda": 1,
        "epsilon": 20,
    }

    output = segment_image(dummy_image, params)

    show_segmentation(dummy_image, output["original_partition"])
    show_segmentation(dummy_image, output["final_partition"])

    plt.show()
