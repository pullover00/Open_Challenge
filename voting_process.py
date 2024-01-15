import cv2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree

""" Find clusters of pointcloud

Author: Tessa Pulli
MatrNr: 
"""

def update_voting_matrix(voting_matrix, filename, final_label):
   
    object_name = filename[:-7]

    # Check if the object is already in the matrix
    for row in voting_matrix:
        if row[0].lower() == object_name.lower():
            # If yes, append the final_label to the existing list
            row.append(final_label)
            break
    else:
        # If the object is not in the matrix, add a new row
        voting_matrix.append([object_name, final_label])

    return voting_matrix

def label_voting(filled_image, labelled_pixel_image, target_keypoints, matches):
    height, width, _ = filled_image.shape

    image_matches = np.zeros((height, width))

    target_points = np.int32([target_keypoints[match.queryIdx].pt for match in matches])
    target_points[:, 0] = np.clip(target_points[:, 0], 0, width - 1)
    target_points[:, 1] = np.clip(target_points[:, 1], 0, height - 1)
    image_matches[target_points[:, 1], target_points[:, 0]] = 1

    for y in range(height):
        for x in range(width):
            if np.array_equal(filled_image[y, x, :], [255, 255, 255]):
                filled_image[y, x, :] = [0, 0, 0]

    for y in range(height):
        for x in range(width):
            if image_matches[y, x] == 1:
                image_matches[y, x] = labelled_pixel_image[y, x]

    unique_labels, label_counts = np.unique(image_matches, return_counts=True)
    label_counts = label_counts.astype(int)
    unique_labels = unique_labels.astype(int)

    total_label_counts = np.bincount(labelled_pixel_image.ravel())

    normalized_label_counts = np.zeros(len(unique_labels))
    for i in range(1, len(unique_labels)):
        normalized_label_counts[i] = label_counts[i] / total_label_counts[unique_labels[i]]

    print("How many labels fell into each cluster?:")
    for label, count in zip(unique_labels, normalized_label_counts):
        print(f"Label: {label}, Count: {count}")

    final_label_index = np.argmax(normalized_label_counts)
    final_label = unique_labels[final_label_index]

    return final_label

def evaluate_voting(voting_matrix):
    return(voting_matrix)