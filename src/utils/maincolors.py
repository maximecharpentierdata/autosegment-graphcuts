from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


def _make_histogram(
    reshaped_image: np.ndarray, n_colors: int = 2, bins: int = 5
) -> Tuple[List[int], np.ndarray]:
    """Fetch top colors from the histogram

    Args:
        reshaped_image (np.ndarray): Image reshaped to (N, 3)
        n_colors (int, optional): Number of colors. Defaults to 2.
        bins (int, optional): Number of bins for the histogram. Defaults to 5.

    Returns:
        Tuple[List[int], np.ndarray]: Unflatten indexes of top colors and all edges
    """
    hist, edges = np.histogramdd(reshaped_image, bins=bins)
    best_indexes = hist.flatten().argsort()[::-1][:n_colors]

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
    reshaped_image: np.ndarray, n_colors: int = 2, bins: int = 5
) -> np.ndarray:
    """Compute best colors

    Args:
        reshaped_image (np.ndarray): Reshaped image to (N, 3)
        n_colors (int, optional): Number of main colors. Defaults to 2.
        bins (int, optional): Bin size. Defaults to 5.

    Returns:
        np.ndarray: Main colors
    """
    unflatten_indexes, edges = _make_histogram(reshaped_image, n_colors, bins)
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
            dict(mu=np.mean(sub_image, axis=0), sigma2=np.var(sub_image, axis=0))
        )
    return distributions


def show_mask(mask: np.ndarray, image: np.ndarray):
    r = np.where(mask, image[..., 0], 255)[..., np.newaxis]
    g = np.where(mask, image[..., 1], 255)[..., np.newaxis]
    b = np.where(mask, image[..., 2], 255)[..., np.newaxis]
    plt.imshow(np.concatenate([r, g, b], axis=-1))
