from typing import Dict, List, Tuple
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def make_original_partition(segmented_image: np.ndarray) -> dict:
    partition = {k: set() for k in range(int(segmented_image.max()) + 1)}
    for n in range(len(segmented_image)):
        label = segmented_image[n]
        partition[label].add(str(n))
    return partition


def compute_distributions(
    segmented_image: np.ndarray, reshaped_image: np.ndarray
) -> List[Dict[str, np.ndarray]]:
    distributions = []
    for k in range(int(segmented_image.max()) + 1):
        sub_image = reshaped_image[segmented_image == k]
        distributions.append(
            dict(
                mu=np.mean(sub_image, axis=0),
                sigma2=np.var(sub_image, axis=0),
                size=len(sub_image),
            )
        )
    return distributions


def segmentation(
    image: np.ndarray, params: dict, verbose: bool = True
) -> Tuple[List[Dict[str, np.ndarray]], dict]:
    if verbose:
        print("Making KMeans clustering...")
    reshaped_image = image.reshape((-1, 3))
    kmeans = MiniBatchKMeans(n_clusters=params["n_clusters"], random_state=0)
    segmented_image = kmeans.fit_predict(reshaped_image)
    distributions = compute_distributions(segmented_image, reshaped_image)
    original_partition = make_original_partition(segmented_image)
    return distributions, original_partition
