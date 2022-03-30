from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm


def _smooth_cost(label_1: int, label_2: int, epsilon: float) -> float:
    if label_1 == label_2:
        return 0
    else:
        return epsilon


def _data_cost(
    pixel: np.ndarray, distributions: List[Dict[str, np.ndarray]]
) -> np.ndarray:
    """Compute data costs between a given pixel and all main colors distributions

    Args:
        pixel (np.ndarray): Candidate pixel
        distributions (dict): Distributions of main colors (with means and variances)

    Returns:
        float: Data costs values
    """
    joined_logprobas = []
    for distribution in distributions:
        mu, sigma2, size = (
            distribution["mu"],
            distribution["sigma2"],
            distribution["size"],
        )
        joined_logprobas.append(
            np.sum(-1 / 2 * np.log(sigma2) - ((pixel - mu) ** 2) / 2 / sigma2)
            + np.log(size)
        )

    # Using exp normalize trick to avoid overflow
    max_l = np.max(joined_logprobas)
    corrected_conditional_logprobas = [l - max_l for l in joined_logprobas]

    # Compute probabilities
    probabilities = np.exp(corrected_conditional_logprobas) / np.sum(
        np.exp(corrected_conditional_logprobas)
    )

    return -np.log(probabilities + 1e-8)


def _get_label(n: int, partition: Dict[int, set]) -> int:
    for label in partition:
        if str(n) in partition[label]:
            return label
    raise ValueError()


def _get_neighboors(
    flattened_index: int, shape: Tuple[int, int]
) -> List[Tuple[int, int]]:
    x, y = np.unravel_index(flattened_index, shape)
    neighboors = []

    if x != shape[0] - 1:
        neighboors.append((x + 1, y))
    if y != shape[1] - 1:
        neighboors.append((x, y + 1))

    return neighboors


def build_base_graph(image: np.ndarray, verbose: float = True) -> nx.Graph:
    """Build base graph with nodes of a given image taking their flatten indexes as node names

    Args:
        image (np.ndarray): Image

    Returns:
        nx.Graph: Base graph of the image (only pixel nodes)
    """
    reshaped_image = image.reshape(-1, 3)

    graph = nx.Graph()
    mapping_pixels = dict()

    for n in tqdm(range(reshaped_image.shape[0]), disable=(not verbose)):
        mapping_pixels[str(n)] = reshaped_image[n]

    graph.add_nodes_from(mapping_pixels.keys())
    return graph


def add_data_edges(
    base_graph: nx.Graph,
    distributions: List[Dict[str, np.ndarray]],
    alpha: int,
    image: np.ndarray,
    partition: Dict[int, set],
    params: dict,
    verbose: float = True,
) -> Tuple[nx.Graph, List[str]]:
    """Add all edges representing data and smoothing costs for a single alpha-expansion iteration

    Args:
        base_graph (nx.Graph): Base graph with pixel nodes only
        distributions (List[Dict[str, np.ndarray]]): Main colors distributions for computing data costs
        alpha (int): Alpha iteration label
        image (np.ndarray): Image
        partition (dict): Current partition
        params (dict): Params (with keys lambda and epsilon of costs computing)

    Returns:
        Tuple[nx.Graph, List[str]]: New graph and list of auxiliary nodes
    """
    graph = base_graph.copy()
    reshaped_image = image.reshape(-1, 3)

    source = -1  # alpha node
    target = -2  # non-alpha node

    graph.add_nodes_from([str(source), str(target)])

    auxiliary_nodes = []
    edges = []
    for index in tqdm(range(reshaped_image.shape[0]), disable=(not verbose)):
        pixel = reshaped_image[index]
        data_costs = params["lambda"] * _data_cost(pixel, distributions)
        edges.append((str(index), str(source), dict(capacity=data_costs[alpha])))

        label = _get_label(index, partition)

        if label == alpha:
            edges.append((str(index), str(target), dict(capacity=np.inf)))
        else:
            edges.append((str(index), str(target), dict(capacity=data_costs[label])))

        neighboors = _get_neighboors(index, image.shape[:-1])

        for i, j in neighboors:
            neighboor_index = np.ravel_multi_index((i, j), image.shape[:-1])
            label_neighboor = _get_label(neighboor_index, partition)

            cost = _smooth_cost(label, label_neighboor, params["epsilon"])
            if label_neighboor != label:
                auxiliary_node = f"auxiliary_{index}_{neighboor_index}"
                auxiliary_nodes.append(auxiliary_node)

                cost_n = _smooth_cost(label, alpha, params["epsilon"])
                edges.append((str(index), auxiliary_node, dict(capacity=cost_n)))
                cost_neighboor = _smooth_cost(label_neighboor, alpha, params["epsilon"])
                edges.append(
                    (
                        auxiliary_node,
                        str(neighboor_index),
                        dict(capacity=cost_neighboor),
                    )
                )
                edges.append((auxiliary_node, str(target), dict(capacity=cost)))
            else:
                cost_n = _smooth_cost(label, alpha, params["epsilon"])
                edges.append((str(index), str(neighboor_index), dict(capacity=cost_n)))

    graph.add_nodes_from(auxiliary_nodes)
    graph.add_edges_from(edges)
    return graph, set(auxiliary_nodes)


def compute_energy(
    distributions: List[Dict[str, np.ndarray]],
    image: np.ndarray,
    partition: Dict[int, set],
    params: dict,
    verbose: bool = True,
):
    reshaped_image = image.reshape(-1, 3)

    data_cost = 0
    smooth_cost = 0

    for n in tqdm(range(reshaped_image.shape[0]), disable=(not verbose)):
        pixel = reshaped_image[n]

        # Data cost update
        data_costs = params["lambda"] * _data_cost(pixel, distributions)
        label = _get_label(n, partition)
        data_cost += data_costs[label]

        # Smoothing cost update
        neighboors = _get_neighboors(n, image.shape[:-1])
        for i, j in neighboors:
            flattened_index = np.ravel_multi_index((i, j), image.shape[:-1])
            label_neighboor = _get_label(flattened_index, partition)
            smooth_cost += _smooth_cost(label, label_neighboor, params["epsilon"])

    return {
        "data_cost": data_cost,
        "smooth_cost": smooth_cost,
    }
