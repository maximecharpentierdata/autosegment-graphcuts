from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt


def make_arbitrary_partition(image: np.ndarray, main_colors: np.ndarray) -> dict:
    """Generates an initial partition of using euclidian distance with main colors

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


def construct_segmentation(
    partition: Dict[int, set], shape: Tuple[int, int]
) -> np.ndarray:
    """Generates segmented image from partition

    Args:
        partition (dict): Partition
        shape (Tuple[int, int]): Shape of the original image

    Returns:
        np.ndarray: Segmented image
    """
    new_image = np.zeros(shape)
    for label in partition:
        for node in partition[label]:
            if int(node) > -1:
                index = np.unravel_index(int(node), shape)
                new_image[index] = label
    return new_image


def show_segmentation(image: np.ndarray, partition: Dict[int, set]):
    shape = image.shape[:-1]
    segmented_image = construct_segmentation(partition, shape)
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes[0].imshow(image)
    axes[0].set_axis_off()
    axes[1].imshow(segmented_image)
    axes[1].set_axis_off()


def make_expansion(
    original_partition: Dict[int, set],
    binary_partition: Tuple[set, set],
    alpha: int,
    auxiliary_nodes: List[str],
) -> Dict[int, str]:
    """Finishes one alpha-expansion iteration from the obtained s-t cut

    Args:
        original_partition (Dict[int, set]): Original partition
        binary_partition (Tuple[set, set]): Binary partition obtained for
                                            the s-t cut, s being alpha and t non-alpha
        alpha (int): Alpha iteration label
        auxiliary_nodes (List[str]): Auxiliary nodes list

    Returns:
        Dict[int, str]: New partition
    """
    binary_partition = [
        partition.difference({"-1", "-2"}.union(auxiliary_nodes))
        for partition in binary_partition
    ]
    final_partition = dict()
    for label in original_partition:
        if label != alpha:
            final_partition[label] = original_partition[label].intersection(
                binary_partition[0]
            )
        else:
            final_partition[label] = original_partition[label].union(
                binary_partition[1]
            )
    return final_partition
