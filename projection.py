import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from camera_params import fx_rgb, fy_rgb, cx_rgb, cy_rgb

def project_3d_to_2d(points_3d, image_size=(640, 480)):
    """
    Project 3D points to 2D image space.

    Parameters:
    - points_3d: Open3D PointCloud object
    - image_size: Tuple (width, height) of the output image

    Returns:
    - image: NumPy array representing the image with drawn points and colors
    """
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

def project_clusters_to_2d(pcd, labels, cmap='jet'):
    # Normalize cluster labels to start from 0
    labels_normalized = labels - labels.min()

    # Map cluster labels to colors using a specified colormap
    colors = plt.get_cmap(cmap)(labels_normalized / (labels_normalized.max() if labels_normalized.max() > 0 else 1))

    # Set the color of points labeled as noise (labels < 0) to black
    colors[labels_normalized < 0] = 0

    # Assign colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Project 3D points to 2D image space
    points_2d = project_3d_to_2d(pcd)

    # Create an empty image
    image_shape = (int(np.max(points_2d[:, 1]) + 1), int(np.max(points_2d[:, 0]) + 1), 3)
    image = np.zeros(image_shape, dtype=np.uint8)

    # Fill the image with cluster colors based on the projected 2D points
    for point, label in zip(points_2d.astype(int), labels_normalized):
        if label >= 0:
            image[point[1], point[0], :] = (255 * colors[label, :3]).astype(np.uint8)
    
    print(image)

    return image
