import numpy as np

def project_3d_to_2d(points_3d, focal_length, optical_center):
    """
    Project 3D points to 2D image space.

    Parameters:
    - points_3d: Nx3 array of 3D points
    - focal_length: Focal length of the camera
    - optical_center: Tuple (cx, cy) representing the optical center coordinates

    Returns:
    - points_2d: Nx2 array of 2D points
    """
    cx, cy = optical_center
    points_2d = np.zeros((points_3d.shape[0], 2))

    for i in range(points_3d.shape[0]):
        X, Y, Z = points_3d[i]
        u = (focal_length * X / Z) + cx
        v = (focal_length * Y / Z) + cy
        points_2d[i] = [u, v]

    return points_2d


