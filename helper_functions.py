#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results and calculate the silhouette score

Author: FILL IN
MatrNr: FILL IN
"""

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import distance
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import cv2
from typing import Tuple, List

def plot_dominant_plane(pcd: o3d.geometry.PointCloud,
                        inliers: np.ndarray,
                        plane_eq: np.ndarray) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    # Filter the inlier points and color them red
    inlier_indices = np.nonzero(inliers)[0]
    inlier_cloud = pcd.select_by_index(inlier_indices)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inlier_indices, invert=True)

    # Create a rotation matrix according to the plane equation.
    # Detailed explanation of the approach can be found here: https://math.stackexchange.com/q/1957132
    normal_vector = -plane_eq[0:3] / np.linalg.norm(plane_eq[0:3])
    u1 = np.cross(normal_vector, [0, 0, 1])
    u2 = np.cross(normal_vector, u1)
    rot_mat = np.c_[u1, u2, normal_vector]

    # Create a coordinate frame and transform it to a point on the plane and with its z-axis in the same direction as
    # the normal vector of the plane
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frame.rotate(rot_mat, center=(0, 0, 0))
    if any(inlier_indices):
        coordinate_frame.translate(np.asarray(inlier_cloud.points)[-1])
        coordinate_frame.scale(0.3, np.asarray(inlier_cloud.points)[-1])

    geometries = [inlier_cloud, outlier_cloud, coordinate_frame]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for p in geometries:
        vis.add_geometry(p)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()

def plot_clustering_results(pcd: o3d.geometry.PointCloud,
                            labels: np.ndarray,
                            method_name: str,
                            cmap: str = "tab20"):
    #labels = labels - labels.min()
    print(method_name + f": Point cloud has {int(labels.max())} clusters")
    colors = plt.get_cmap(cmap)(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

# From https://github.com/isl-org/Open3D/issues/2
def text_3d(text, pos, direction=None, degree=0.0, font='RobotoMono-Medium.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd

def crop_image(img):
    height = img.shape[0]
    width = img.shape[1]

    # Checking image is grayscale or not. If image shape is 2 then gray scale otherwise not
    if len(img.shape) == 2:
        gray_input_image = img.copy()
    else:
        # Converting BGR image to grayscale image
        gray_input_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # To find upper threshold, we need to apply Otsu's thresholding
    upper_threshold, thresh_input_image = cv2.threshold(
        gray_input_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Calculate lower threshold
    lower_threshold = 0.5 * upper_threshold

    # Apply canny edge detection
    canny = cv2.Canny(img, lower_threshold, upper_threshold)
    # Finding the non-zero points of canny
    pts = np.argwhere(canny > 0)

    # Finding the min and max points
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    # Crop ROI from the givn image
    output_image = img[y1:y2, x1:x2]

    return(output_image)

def filter_matches(matches: Tuple[Tuple[cv2.DMatch]]) -> List[cv2.DMatch]:
    """Filter out all matches that do not satisfy the Lowe Distance Ratio Condition

    :param matches: Holds all the possible matches. Each 'row' are matches of one source_keypoint to target_keypoint
    :type matches: Tuple of tuples of cv2.DMatch https://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html

    :return filtered_matches: A list of all matches that fulfill the Low Distance Ratio Condition
    :rtype: List[cv2.DMatch]
    """
    ######################################################
    # Initialize filtered matches
    filtered_matches = []

    for match in matches:
        # Find closest match
        m, n = match
        if m.distance < 0.9 * n.distance:
            filtered_matches.append(m) # Append to list if condition is fulfilled
        
    return filtered_matches

def show_image(img: np.ndarray, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.ndarray with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.

    elif img.dtype == np.float64:
        img = img.astype(np.float32)

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA) * 255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)

def draw_rectangles(scene_img: np.ndarray,
                    object_img: np.ndarray,
                    homography: np.ndarray) -> np.ndarray:
    """Plot rectangles with size of object_img into scene_img given the homography transformation matrix

    :param scene_img: Image to draw rectangles into
    :type scene_img: np.ndarray with shape (height, width, channels)

    :param object_img: Image of the searched object which defines the size of the rectangles before transformation
    :type object_img: np.ndarray with shape (height, width, channels)

    :param homography: Projective Transformation matrix for homogeneous coordinates
    :type homography: np.ndarray with shape (3, 3)

    :return: Copied image of scene_img with rectangle drawn on top
    :rtype: np.ndarray with  the same shape (height, width, channels) as scene_img
    """
    output_img = scene_img.copy()

    # Get the height and width of our template object which will define the size of the rectangles we draw
    height, width = object_img.shape[0:2]

    # Define a rectangle with the 4 vertices. With the top left vertex at position [0,0]
    rectangle = np.array([[0, 0],
                          [width, 0],
                          [width, height],
                          [0, height]], dtype=np.float32)

    # Add ones for homogeneous transform
    hom_point = np.c_[rectangle, np.ones(rectangle.shape[0])]

    # Use homography to transform the rectangle accordingly
    rectangle_tf = (homography @ hom_point.T).T
    rectangle_tf = np.around((rectangle_tf[..., 0:2].T/rectangle_tf[..., 2]).T).astype(np.int32)

    cv2.polylines(output_img, [rectangle_tf], isClosed=True, color=(0, 255, 0), thickness=3)

    # Change the top line to be blue, so we can tell the top of the object
    cv2.line(output_img, tuple(rectangle_tf[0]), tuple(rectangle_tf[1]), color=(255, 0, 0), thickness=3)

    return output_img

def label_objects(img, labeled_pixels):
    # Create an RGB image for visualization

    #_, binary_image = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        # Convert to 8-bit binary image
    binary_image = (labeled_pixels > 0).astype(np.uint8) * 255


    # Find contours based on the input labels
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)  # Draw contour for all objects in white

    # Iterate over the contours and find the label at the center
    for contour in contours:
        # Find the centroid of each contour
        centroid = cv2.moments(contour)
        if centroid["m00"] != 0:
            cx = int(centroid["m10"] / centroid["m00"])
            cy = int(centroid["m01"] / centroid["m00"])
        else:
            cx, cy = 0, 0

        # Determine the label at the center of the object
        label_at_center = labeled_pixels[cy, cx]

        # Display the cluster label at the centroid
        cv2.putText(img, f'Cluster {label_at_center}', (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the labeled image
        cv2.imshow('Labeled Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def labeling_pixels(filled_image: np.array)-> np.array:
            """
            filled_image: Image with filled holes in the shape of (height, width, 3) -> rgb image

            r: labeled_image: height, width, 1 -> returning all the given labels 
            """
            height, width, _ = filled_image.shape

            # Convert 1 to filled_image values
            for y in range(height):
                for x in range(width):
                    if np.array_equal(filled_image[y, x, :], [255, 255, 255]):
                        filled_image[y, x, :] = [0, 0, 0]

            labeled_image = np.zeros((height, width))

            # Reshape the array to a 2D array where each row represents a pixel and each column represents an RGB value
            pixels = filled_image.reshape((-1, 3))

            # Get unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # Display the count of unique colors
            print("Number of unique colors:", len(unique_colors))

            # Display unique colors
            print("Unique colors:")
            for color in unique_colors:
                print(color)

            for y in range(height):
                for x in range(width):
                     label_color = 0
                     for color in unique_colors:
                        if np.array_equal(filled_image[y, x, :], color):
                            labeled_image[y, x] = label_color
                        label_color += 1 
                        
            return (labeled_image.astype(int))
