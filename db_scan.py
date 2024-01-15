#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree

""" Find clusters of pointcloud

Author: Tessa Pulli
MatrNr: 
"""

def dbscan(points: np.ndarray,
           eps: float = 0.05,
           min_samples: int = 10) -> np.ndarray:
    """ Find clusters in the provided data coming from a pointcloud using the DBSCAN algorithm.

    The algorithm was proposed in Ester, Martin, et al. "A density-based algorithm for discovering clusters in large
    spatial databases with noise." kdd. Vol. 96. No. 34. 1996.

    :param points: The (down-sampled) points of the pointcloud to be clustered
    :type points: np.ndarray with shape=(n_points, 3)

    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :type eps: float

    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core
        point. This includes the point itself.
    :type min_samples: float

    :return: Labels array with a different label for each cluster for each point (shape = (n_points,) and dtype=int)
            The label -1 is assigned to points that are considered to be noise.
    :rtype: np.ndarray
    """

    labels = np.zeros(len(points), dtype=int)
    C = 0

    for data_point in range(len(points)):
        if labels[data_point] != 0:
            continue

        # Calculate distances using vectorized operations
        distances = np.linalg.norm(points - points[data_point], axis=1)

        # Find neighbors within epsilon distance
        neighbors = np.where(distances < eps)[0]

        if len(neighbors) < min_samples:
            labels[data_point] = -1
        else:
            C += 1
            labels[data_point] = C
            i = 0
            while i < len(neighbors):
                Pn = neighbors[i]
                if labels[Pn] == -1 or labels[Pn] == 0:
                    labels[Pn] = C

                    # Calculate distances for the new neighbors using vectorized operations
                    pt_distances = np.linalg.norm(points - points[Pn], axis=1)

                    # Find new neighbors within epsilon distance
                    pt_neighbors = np.where(pt_distances < eps)[0]

                    if len(pt_neighbors) >= min_samples:
                        neighbors = np.concatenate((neighbors, pt_neighbors))

                i += 1

    min_cluster_size = 150 
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Create a mapping from old labels to new labels
    label_mapping = {label: int(i) + 1 for i, label in enumerate(unique_labels) if label != -1}

    # Print original labels and counts
#    print("Original labels and counts:")
#    for label, count in zip(unique_labels, label_counts):
#        print(f"Cluster: {label}, Count: {count}")

    # Eliminate small clusters
    for label, count in zip(unique_labels, label_counts):
        if count < min_cluster_size and label != -1:
            labels[labels == label] = -1

    # Relabel clusters
    # Create a mapping from old labels to new labels
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    new_label = 1
    for label in unique_labels:
        if label != -1:
            labels[labels == label] = new_label
            new_label += 1

    # Print modified labels and counts
#    print("Modified labels and counts:")
#    unique_labels, label_counts = np.unique(labels.astype(int), return_counts=True)
#    for label, count in zip(unique_labels, label_counts):
#        print(f"Label: {label}, Count: {count}")

    return labels #.astype(int)
