#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
    # 1. Initially set label for each point to 0
    labels = np.zeros(len(points), dtype=int)

    # C for labeling clusters
    C = 0

    for data_point in range(len(points)):
        if labels[data_point] != 0:
            continue

        # 2. Retrieve points in an epsilon neighborhood (Region query)
        neighbors = [region_point for region_point in range(len(points)) if np.linalg.norm(points[region_point] - points[data_point]) < eps]

        # 3. If the number of points is smaller than min_samples, points are marked as noise
        if len(neighbors) < min_samples:
            labels[data_point] = -1

        # 4. If the number of points is bigger than min_samples, xi is a core point
        else:
            C += 1  # Cluster name
            labels[data_point] = C  # Data label with cluster name
            i = 0
            while i < len(neighbors):  # Loop over neighboring points
                Pn = neighbors[i]
                if labels[Pn] == -1 or labels[Pn] == 0:  # if point is noise or unlabeled, give it a label
                    labels[Pn] = C

                    # Region query (within the loop)
                    pt_neighbors = [region_point for region_point in range(len(points)) if np.linalg.norm(points[region_point] - points[Pn]) < eps]

                    if len(pt_neighbors) >= min_samples:
                        neighbors.extend(pt_neighbors)

                i += 1

    # Post-processing: Eliminate small clusters
    min_cluster_size = 50 

    unique_values, counts = np.unique(labels, return_counts=True)
    unique_count = len(unique_values)

    unique_labels, label_counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, label_counts):
        if count < min_cluster_size:
            labels[labels == label] = -1
    
    return labels