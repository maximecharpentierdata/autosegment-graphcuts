from typing import Dict, List, Tuple
from black import main

import matplotlib.pyplot as plt
import numpy as np


def _make_histogram(
    reshaped_image: np.ndarray, threshold: float, bins: int = 5
) -> Tuple[List[int], np.ndarray]:
    """Fetch top colors from the histogram

    Args:
        reshaped_image (np.ndarray): Image reshaped to (N, 3)
        n_colors (int, optional): Number of colors. Defaults to 2.
        bins (int, optional): Number of bins for the histogram. Defaults to 5.

    Returns:
        Tuple[List[int], np.ndarray]: Unflatten indexes of top colors and all edges
    """
    ranges = (
        (np.min(reshaped_image[:, 0]), np.max(reshaped_image[:, 0])),
        (np.min(reshaped_image[:, 1]), np.max(reshaped_image[:, 1])),
        (np.min(reshaped_image[:, 2]), np.max(reshaped_image[:, 2])),
    )
    hist, edges = np.histogramdd(reshaped_image, bins=bins, range=ranges)
    hist = hist / len(reshaped_image)
    best_indexes = np.where(hist.flatten() > threshold)[0]

    unflatten_indexes = [np.unravel_index(index, hist.shape) for index in best_indexes]
    return unflatten_indexes, np.array(edges)


def _get_best_color_boundaries(
    dimension_indexes: Tuple[int, int, int], edges: np.ndarray
) -> List[Tuple[float, float]]:
    """Compute boundaries for each dimension of a main color

    Args:
        dimension_indexes (int): Indexes of main color (R, G, B)
        edges (np.ndarray): Edges from the histogram computation

    Returns:
        List[float, float]: List of the boundaries of the main color
    """
    boundaries = []
    for n, dimension_index in enumerate(dimension_indexes):
        boundaries.append((edges[n, dimension_index], edges[n, dimension_index + 1]))
    return boundaries


def _get_best_colors_boundaries(
    indexes: List[Tuple[int, int, int]], edges: np.ndarray
) -> np.ndarray:
    """Compute boundaries of all main colors for all dimension

    Args:
        indexes (List[Tuple[int, int, int]]): List of color indexes (RGB)
        edges (np.ndarray): Edges from the histogram computation

    Returns:
        np.ndarray: Boundaries of each colors shape (N, 3, 2)
    """
    main_colors = []
    for index in indexes:
        main_colors.append(_get_best_color_boundaries(index, edges))
    return np.array(main_colors)


def get_main_colors(
    reshaped_image: np.ndarray, threshold: float, bins: int = 5
) -> np.ndarray:
    """Compute best colors

    Args:
        reshaped_image (np.ndarray): Reshaped image to (N, 3)
        n_colors (int, optional): Number of main colors. Defaults to 2.
        bins (int, optional): Bin size. Defaults to 5.

    Returns:
        np.ndarray: Main colors
    """
    unflatten_indexes, edges = _make_histogram(reshaped_image, threshold, bins)
    main_colors = _get_best_colors_boundaries(unflatten_indexes, edges)
    return main_colors


def show_main_colors(main_colors: np.ndarray):
    """Plot main colors

    Args:
        main_colors (np.ndarray): Main colors
    """
    _, axes = plt.subplots(1, len(main_colors), figsize=(15, 2))
    for k in range(len(axes)):
        axes[k].imshow([[np.mean(main_colors, axis=2)[k] / 256]])
        axes[k].set_axis_off()


def compute_masks(main_colors: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Compute masks for main_colors segmentation

    Args:
        main_colors (np.ndarray): Main colors
        image (np.ndarray): Image

    Returns:
        np.ndarray: Masks
    """
    masks = []
    for main_color in main_colors:
        condition = (image[..., 0] > main_color[0, 0]) & (
            image[..., 0] < main_color[0, 1]
        )
        condition = (
            condition
            & (image[..., 1] > main_color[1, 0])
            & (image[..., 1] < main_color[1, 1])
        )
        condition = (
            condition
            & (image[..., 2] > main_color[2, 0])
            & (image[..., 2] < main_color[2, 1])
        )
        masks.append(condition)
    return masks


def compute_distributions(
    masks: np.ndarray, image: np.ndarray
) -> List[Dict[str, np.ndarray]]:
    """Compute distributions (mean, variance) for each main color

    Args:
        masks (np.ndarray): Masks of each main color
        image (np.ndarray): Image

    Returns:
        List[Dict[str, np.ndarray]]: Distributions of main colors with
                                     keys mu (for mean) and sigma2 (for variance)
    """
    distributions = []
    for mask in masks:
        sub_image = image[mask]
        distributions.append(
            dict(
                mu=np.mean(sub_image, axis=0),
                sigma2=np.var(sub_image, axis=0),
                size=len(sub_image),
            )
        )
    return distributions


def show_mask(mask: np.ndarray, image: np.ndarray):
    r = np.where(mask, image[..., 0], 255)[..., np.newaxis]
    g = np.where(mask, image[..., 1], 255)[..., np.newaxis]
    b = np.where(mask, image[..., 2], 255)[..., np.newaxis]
    plt.imshow(np.concatenate([r, g, b], axis=-1))


def make_original_partition(image: np.ndarray, main_colors: np.ndarray) -> dict:
    """Generate an initial partition of using euclidian distance with main colors

    Args:
        image (np.ndarray): Image
        main_colors (np.ndarray): Main colors boundaries

    Returns:
        dict: Initial partition
    """
    partition = {k: set() for k in range(len(main_colors))}
    reshaped_image = image.reshape(-1, 3)
    for n in range(len(reshaped_image)):
        pixel = reshaped_image[n]
        distances = np.linalg.norm(pixel - np.mean(main_colors, 2), axis=1)
        label = np.argmin(distances)
        partition[label].add(str(n))
    return partition


def segmentation(
    image: np.ndarray, params: dict, verbose: bool = True, threshold: float = 0.1
) -> Tuple[List[Dict[str, np.ndarray]], dict]:
    """Run the original segmentation using histogram of colors

    Args:
        image (np.ndarray): Image
        params (dict): Params with key bins (int)
        verbose (bool, optional): Verbose. Defaults to True.
        threshold (float, optional): Threshold. Defaults to 0.1.

    Returns:
        Tuple[List[Dict[str, np.ndarray]], dict]: Distributions of main colors
        and original partition
    """
    if verbose:
        print("Computing main colors...")
    reshaped_image = image.reshape((-1, 3))
    main_colors = get_main_colors(
        reshaped_image, bins=params["bins"], threshold=threshold
    )
    while len(main_colors) == 0:
        threshold = threshold * 9 / 10
        main_colors = get_main_colors(
            reshaped_image, bins=params["bins"], threshold=threshold
        )
    masks = compute_masks(main_colors, image)
    distributions = compute_distributions(masks, image)
    original_partition = make_original_partition(image, main_colors)
    return distributions, original_partition
