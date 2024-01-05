import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb

def project_3d_to_2d(points_3d):
    """
    Project 3D points to 2D image space.

    Parameters:
    - points_3d: Open3D PointCloud object
    - image_size: Tuple (width, height) of the output image

    Returns:
    - image: NumPy array representing the image with drawn points and colors
    """
    image_size = (640, 480)

    # Convert Open3D PointCloud to NumPy array
    points = np.asarray(points_3d.points)

    num_points = len(points)
    points_2d = np.zeros((num_points, 2))
    
    # Reverse the order of color channels (BGR to RGB)
    pt_colors = np.asarray(points_3d.colors)[..., ::-1]*255

    for i in range(num_points):
        X, Y, Z = points[i]
        u = (fx_rgb * X / Z) + cx_rgb
        v = (fy_rgb * Y / Z) + cy_rgb

        # Check for NaN in u or v
        if not np.isnan(u) and not np.isnan(v):
            points_2d[i] = [u, v]

    # Create a black image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # Draw circles for each point with its associated color
    for point, color in zip(points_2d.astype(int), pt_colors):
        cv2.circle(image, tuple(point), 1, color, -1)

    return image


