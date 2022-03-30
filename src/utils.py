import numpy as np
import matplotlib.pyplot as plt
import itertools
import cv2


def read_image(
    frame: str,
    path: str = "./data/VOCdevkit/VOC2007/JPEGImages/",
    extension: str = ".jpg",
) -> np.ndarray:
    frame = frame.zfill(6)
    return plt.imread(path + frame + extension)


def resize_image(image: np.ndarray, resize: bool = True):
    if resize:
        max_index = np.argmax(image.shape[:2])
        min_index = np.argmin(image.shape[:2])
        if min_index == max_index:
            max_index += int(not min_index)
        max_value = 100
        min_value = image.shape[min_index] * max_value / image.shape[max_index]

        new_shape = [0, 0]
        new_shape[min_index] = int(min_value)
        new_shape[max_index] = int(max_value)

        resized_image = cv2.resize(image, new_shape[::-1])
    else:
        resized_image = image
    return resized_image


def iou(binary_mask: np.ndarray, label: np.ndarray):
    tp = np.sum(np.logical_and(binary_mask, label))
    fp = np.sum(np.logical_and(binary_mask, np.logical_not(label)))
    fn = np.sum(np.logical_and(np.logical_not(binary_mask), label))
    return tp / (tp + fp + fn)


def evaluate_proposal(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    iou_scores = []
    for i in range(predictions.shape[0]):
        iou_scores.append(iou(predictions[i], labels[i]))
    return np.mean(iou_scores)


def evaluate(
    segmented_image: np.ndarray,
    label: np.ndarray,
):
    values_segmented_image = np.unique(segmented_image)
    values_label = np.unique(label)

    mappings = []
    if len(values_segmented_image) > len(values_label):
        for permutation in itertools.permutations(values_segmented_image):
            mappings.append(dict(zip(values_label, permutation)))
    else:
        for permutation in itertools.permutations(values_label):
            mappings.append(dict(zip(permutation, values_segmented_image)))

    mean_ious = []
    for mapping in mappings:
        predictions = []
        labels = []
        for k, value in mapping.items():
            predictions.append(segmented_image == value)
            labels.append(label == k)

        mean_ious.append(evaluate_proposal(np.array(predictions), np.array(labels)))
    return np.max(mean_ious)
