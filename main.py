#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision (376.081)
Exercise 5: Open Challenge
Automation & Control Institute, TU Wien

Tessa Pulli
Matrikel Nummer:

"""

from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from fit_plane import fit_plane
from helper_functions import *
from 2d_to_3d import *

if __name__ == '__main__':

    pcd_path = '/image000'

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.01
    uniform_every_k_points = 10

    debug_output = True

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/")) + str(pcd_path) + ".pcd",
                                  remove_nan_points=True, remove_infinite_points=True)
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    if use_voxel_downsampling:
        pcd_sampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = pcd.uniform_down_sample(uniform_every_k_points)

    # Apply your own plane-fitting algorithm
    plane_model, best_inliers = fit_plane(pcd=pcd_sampled,
                                         confidence=confidence,
                                         inlier_threshold=inlier_threshold)
    inlier_indices = np.nonzero(best_inliers)[0]

    # Alternatively use the built-in function of Open3D
    # plane_model, inlier_indices = pcd_sampled.segment_plane(distance_threshold=inlier_threshold,
    #                                                        ransac_n=3,
    #                                                        num_iterations=500)
    
    # Convert the inlier indices to a Boolean mask for the pointcloud
    best_inliers = np.full(shape=len(pcd_sampled.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd_sampled.select_by_index(inlier_indices, invert=True)

    # Plot detected plane and remaining pointcloud
    if debug_output:
        plot_dominant_plane(pcd_sampled, best_inliers, plane_model)
        o3d.visualization.draw_geometries([scene_pcd])

    # Convert to NumPy array
    points = np.asarray(scene_pcd.points, dtype=np.float32)

    #  3D to 2D 
    dimension_change(pcd_sampled)