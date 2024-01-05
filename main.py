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
import cv2
import os

from fit_plane import fit_plane
from helper_functions import *
from projection import *
from db_scan import *

if __name__ == '__main__':

    # Point cloud path
    pcd_path = '/image000'

    # RANSAC parameters:
    confidence = 0.85
    inlier_threshold = 0.015

    # Downsampling parameters:
    use_voxel_downsampling = True
    voxel_size = 0.005
    uniform_every_k_points = 10

    # DB Scan parameters
    dbscan_eps = 0.05 # start: 0.05
    dbscan_min_points = 15 # start: 15
    
    Plane_extraction = False # show extracted plane
    Projections = False # show 2D projection of point cloud
    DB_Scan = False 
    SIFT = True 

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/")) + str(pcd_path) + ".pcd",
                                  remove_nan_points=True, remove_infinite_points=True)
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Apply your own plane-fitting algorithm
    #plane_model, best_inliers = fit_plane(pcd=pcd_sampled,
    #                                     confidence=confidence,
    #                                     inlier_threshold=inlier_threshold)
    # inlier_indices = np.nonzero(best_inliers)[0]

    # Alternatively use the built-in function of Open3D
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=inlier_threshold,
                                                            ransac_n=3,
                                                            num_iterations=500)
    
    # Convert the inlier indices to a Boolean mask for the pointcloud
    best_inliers = np.full(shape=len(pcd.points, ), fill_value=False, dtype=bool)
    best_inliers[inlier_indices] = True

    # Store points without plane in scene_pcd
    scene_pcd = pcd.select_by_index(inlier_indices, invert=True)

    # Plot detected plane and remaining pointcloud
    if Plane_extraction:
        plot_dominant_plane(pcd, best_inliers, plane_model)
        o3d.visualization.draw_geometries([scene_pcd])

    # 3D to 2D projection        
    result_image = project_3d_to_2d(scene_pcd)

    if Projections:
        cv2.imshow('2D Projection with Colors', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Subsampling point cloud
    if use_voxel_downsampling:
        pcd_sampled = scene_pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        pcd_sampled = scene_pcd.uniform_down_sample(uniform_every_k_points)

    # Convert to NumPy array
    points = np.asarray(pcd_sampled.points, dtype=np.float32)
   
    if DB_Scan:
        labels = dbscan(points,
                        eps=dbscan_eps,
                        min_samples=dbscan_min_points)

        plot_clustering_results(pcd_sampled,
                                labels,
                                "DBSCAN",
                                cmap="tab10")
        
        # Project clusters to 2D space 
        cluster_image = project_3d_to_2d(pcd_sampled)

        # Plot clustered result
        cv2.imshow('2D Projection of Clusters', cluster_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Fill holes
        kernel = np.ones((6, 6), np.uint8)  
        filled_image = cv2.morphologyEx(cluster_image, cv2.MORPH_CLOSE, kernel)
        
        # Plot clustered result
        cv2.imshow('2D Projection of Clusters with closed holes', filled_image)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Label objects

        # Find contours in the clustered image
        contours, _ = cv2.findContours(filled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and add text annotations
        for i, contour in enumerate(contours):
            # Find the centroid of each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Display the cluster label at the centroid
            cv2.putText(filled_image, f'Cluster {i+1}', (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the original labeled image and the clustered image with annotations
        cv2.imshow('Original Labeled Image', result_image)
        cv2.imshow('Clustered Image with Annotations', filled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if SIFT:
        # Find training point cloud
        current_path = Path(__file__).parent
        dir = current_path.joinpath("training/")
        print(dir)
        directory = dir  # Use dir directly, without os.fsencode

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            file_path = os.path.join(directory, filename)
            
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                raise FileNotFoundError("Couldn't load point cloud in " + str(file_path))
            
            # Assuming you have a function project_3d_to_2d defined elsewhere in your code
            projected_image = project_3d_to_2d(pcd)
            
            # Assuming you have a variable filled_image defined elsewhere in your code
            cv2.imshow('2D Projection of Clusters with closed holes', projected_image)   
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                    

            
            
