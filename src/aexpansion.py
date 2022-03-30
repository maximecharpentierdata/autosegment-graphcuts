from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def construct_segmentation(
    partition: Dict[int, set], shape: Tuple[int, int]
) -> np.ndarray:
    """Generate segmented image from partition

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
    """Finishe one alpha-expansion iteration from the obtained s-t cut

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
