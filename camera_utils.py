import json
import re
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

import trimesh


def get_camera_intrinsic_matrix(
    input_json: str | Path,
    camera_id: str = "  1920 1080 brown 0.85",
) -> np.ndarray:
    """
    Read the cameras.json file and return the camera intrinsic matrix K.

    Parameters:
    -----------
    input_json : str | Path
        The path to the cameras.json file
    camera_id : str
        The ID of the camera in the cameras.json file. Default is "  1920 1080 brown 0.85"

    Returns:
    --------
    K : numpy.ndarray
        The 3x3 camera intrinsic matrix
    """
    # Read the cameras.json file
    with open(input_json, "r") as f:
        cameras = json.load(f)

    # Get the camera parameters
    camera = cameras[camera_id]
    fx = camera["focal_x"]
    fy = camera["focal_y"]
    cx = camera["c_x"]
    cy = camera["c_y"]

    # Construct the intrinsic matrix K
    # Following the notation from https://ksimek.github.io/2013/08/13/intrinsic/
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K


def get_camera_distortion_coeffs(
    input_json: str | Path,
    camera_id: str = "  1920 1080 brown 0.85",
) -> Dict[str, float]:
    """
    Read the cameras.json file and return the Brown-Conrady distortion coefficients.

    Parameters:
    -----------
    input_json : str | Path
        The path to the cameras.json file
    camera_id : str
        The ID of the camera in the cameras.json file. Default is "  1920 1080 brown 0.85"

    Returns:
    --------
    distortion : Dict[str, float]
        Dictionary containing the Brown-Conrady distortion coefficients:
        - k1, k2, k3: Radial distortion coefficients
        - p1, p2: Tangential distortion coefficients

    Examples
    --------
    # Use opencv to correct a single point

    # Get the camera intrinsic matrix
    K = get_camera_intrinsic_matrix("cameras.json")

    # Get the camera distortion coefficients
    distortion = get_camera_distortion_coeffs("cameras.json")

    # OpenCV expects coefficients in order: k1, k2, p1, p2, k3
    dist_coeffs = np.array([
        distortion["k1"],
        distortion["k2"],
        distortion["p1"],
        distortion["p2"],
        distortion["k3"]
    ])

    # Point in pixel coordinates
    point = np.array([[100, 200]], dtype=np.float32)

    # Correct the point
    corrected_point = cv2.undistortPoints(
        point,
        K,
        dist_coeffs,
        P=K  # Use same camera matrix for output
    )

    # Example: Correct an entire image
    distorted_image = cv2.imread("distorted.jpg")
    corrected_image = cv2.undistort(
        distorted_image,
        K,
        dist_coeffs
    )
    ```
    """
    # Read the cameras.json file
    with open(input_json, "r") as f:
        cameras = json.load(f)

    # Get the camera parameters
    camera = cameras[camera_id]

    # Extract Brown-Conrady distortion coefficients
    distortion = {
        "k1": camera["k1"],  # Radial distortion coefficient
        "k2": camera["k2"],  # Radial distortion coefficient
        "k3": camera["k3"],  # Radial distortion coefficient
        "p1": camera["p1"],  # Tangential distortion coefficient
        "p2": camera["p2"],  # Tangential distortion coefficient
    }

    return distortion


def axis_angle_to_rotation_matrix(axis_angle: list[float]) -> np.ndarray:
    """
    Convert axis-angle representation to a 3x3 rotation matrix using Rodrigues' rotation formula.

    Parameters:
    -----------
    axis_angle : list[float]
        A 3D vector representing the rotation axis (normalized)
        multiplied by the rotation angle in radians

    Returns:
    --------
    R : np.ndarray
        A 3x3 rotation matrix

    Notes:
    ------
    The axis-angle representation encodes a rotation as a 3D vector where:
    - The direction of the vector represents the rotation axis
    - The magnitude of the vector represents the rotation angle in radians
    """
    angle = np.linalg.norm(axis_angle)
    if angle == 0:
        return np.eye(3)

    axis = np.array(axis_angle) / angle
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def get_camera_poses(input_json: str | Path) -> Dict[int, np.ndarray]:
    """
    Read a GeoJSON file containing camera poses and return
    the camera pose matrices in homogeneous coordinates.

    Parameters:
    -----------
    input_json : str | Path
        The path to the GeoJSON file containing camera poses

    Returns:
    --------
    camera_poses : Dict[int, np.ndarray]
        Dictionary mapping frame numbers to 4x4 homogeneous camera pose matrices
        Each matrix is of the form [R|t] where R is the 3x3 rotation matrix and t is the 3x1 translation vector
        The camera pose matrix transforms points from world coordinates to camera coordinates

    Examples
    --------
    >>> poses = get_camera_poses("path/to/shots.geojson")
    >>> poses[6040]  # Get the 4x4 camera pose matrix for frame 6040
    array([[ 0.999, -0.001,  0.001, -0.988],
           [ 0.001,  0.999,  0.001, -0.981],
           [-0.001, -0.001,  1.000,  0.037],
           [ 0.000,  0.000,  0.000,  1.000]])
    """
    # Read the GeoJSON file
    with open(input_json, "r") as f:
        data = json.load(f)

    camera_poses = {}
    for feature in data["features"]:
        # Extract frame number from filename
        # (searches for one or more zeros followed by one or more digits,
        # followed by ".png")
        filename = feature["properties"]["filename"]
        match = re.search(r"0+(\d+)\.png", filename)
        if not match:
            raise ValueError(
                f"Could not extract frame number from filename: {filename}"
            )
        frame_num = int(match.group(1))

        # Get rotation and translation
        rotation = feature["properties"]["rotation"]  # axis-angle
        translation = feature["properties"][
            "translation"
        ]  # vector from world origin to camera origin

        # Convert axis-angle to rotation matrix
        R = axis_angle_to_rotation_matrix(rotation)

        # Create homogeneous camera pose matrix [R|t]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = translation

        camera_poses[frame_num] = pose

    return camera_poses


def get_camera_extrinsic_matrix(camera_poses: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Convert camera pose matrices to extrinsic matrices.

    The camera pose matrix describes the camera's position and orientation in world coordinates,
    while the extrinsic matrix describes how to transform points from world coordinates to camera coordinates.
    The extrinsic matrix is the inverse of the camera pose matrix.

    Parameters:
    -----------
    camera_poses : Dict[int, np.ndarray]
        Dictionary mapping frame numbers to 4x4 homogeneous camera pose matrices
        Each matrix is of the form [R_c|C] where:
        - R_c is the 3x3 rotation matrix describing camera orientation
        - C is the 3x1 translation vector describing camera position

    Returns:
    --------
    extrinsics : Dict[int, np.ndarray]
        Dictionary mapping frame numbers to 4x4 homogeneous extrinsic matrices
        Each matrix is of the form [R|t] where:
        - R = R_c^T (transpose of camera rotation)
        - t = -R_c^T * C (negative rotated camera position)

    Notes:
    ------
    Based on the derivation from https://ksimek.github.io/2012/08/22/extrinsic/:
    The extrinsic matrix is obtained by inverting the camera pose matrix:
    [R|t] = [R_c|C]^(-1)

    This can be decomposed into:
    R = R_c^T
    t = -R_c^T * C

    Examples
    --------
    >>> poses = get_camera_poses("path/to/shots.geojson")
    >>> extrinsics = get_camera_extrinsics(poses)
    >>> extrinsics[6040]  # Get the 4x4 extrinsic matrix for frame 6040
    array([[ 0.999,  0.001, -0.001,  0.988],
           [-0.001,  0.999,  0.001,  0.981],
           [ 0.001, -0.001,  1.000, -0.037],
           [ 0.000,  0.000,  0.000,  1.000]])
    """
    camera_extrinsics = {}

    for frame_num, pose in camera_poses.items():
        # Extract rotation and translation from camera pose
        R_c = pose[:3, :3]  # Camera rotation matrix
        C = pose[:3, 3]  # Camera position

        # Compute extrinsic matrix components
        R = R_c.T  # transpose of camera rotation
        t = -R @ C  # negative position of camera origin in world coordinates

        # Create homogeneous extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        camera_extrinsics[frame_num] = extrinsic

    return camera_extrinsics


def image_to_world_points(
    image_points: np.ndarray,
    frame_num: int,
    camera_intrinsics: np.ndarray,
    camera_extrinsics: Dict[int, np.ndarray],
    # depth: float = 1.0,
    mesh: trimesh.Trimesh | None = None,
    distortion: Dict[str, float] | None = None,
) -> np.ndarray:
    """
    Convert points from image coordinates to world coordinates.

    Parameters:
    -----------
    image_points : np.ndarray
        Nx2 array of points in image coordinates (pixel coordinates)
    frame_num : int
        Frame number to use for camera pose
    camera_intrinsics : np.ndarray
        3x3 camera intrinsic matrix
    camera_extrinsics : Dict[int, np.ndarray]
        Dictionary mapping frame numbers to 4x4 homogeneous extrinsic matrices
    depth : float
        Depth value to use for back-projection. Default is 1.0
    distortion : Dict[str, float] | None
        Optional dictionary of distortion coefficients from get_camera_distortion_coeffs.
        If None, no distortion correction is applied.

    Returns:
    --------
    world_points : np.ndarray
        Nx3 array of points in world coordinates

    Examples
    --------
    >>> # Get camera parameters
    >>> K = get_camera_intrinsic_matrix("path/to/cameras.json")
    >>> distortion = get_camera_distortion_coeffs("path/to/cameras.json")
    >>> poses = get_camera_poses("path/to/shots.geojson")
    >>> extrinsics = get_camera_extrinsics(poses)

    >>> # Convert a single point with distortion
    >>> image_point = np.array([[960, 540]])  # Center of 1920x1080 image
    >>> world_point = image_to_world_points(
    ...     image_point,
    ...     frame_num=6040,
    ...     camera_intrinsics=K,
    ...     camera_extrinsics=extrinsics,
    ...     distortion=distortion
    ... )
    >>> print(world_point)
    [[-7.489, -0.001, 0.037]]

    >>> # Convert without distortion
    >>> world_point = image_to_world_points(
    ...     image_point,
    ...     frame_num=6040,
    ...     camera_intrinsics=K,
    ...     camera_extrinsics=extrinsics
    ... )
    """
    if frame_num not in camera_extrinsics:
        raise ValueError(f"Frame {frame_num} not found in camera_extrinsics")

    if distortion is not None:
        # Get distortion coefficients in OpenCV format
        dist_coeffs = np.array(
            [
                distortion["k1"],
                distortion["k2"],
                distortion["p1"],
                distortion["p2"],
                distortion["k3"],
            ]
        )

        # Undistort points
        image_points = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            camera_intrinsics,
            dist_coeffs,
            P=camera_intrinsics,  # to return in pixel coordinates?
        ).reshape(-1, 2)
    # else:
    # # Convert to normalized image coordinates without distortion
    # # Subtract principal point (cx, cy) and divide by focal length to get normalized coords
    # # camera_intrinsics[:2, 2] gets [cx, cy] from the K matrix
    # # camera_intrinsics[0, 0] gets focal length fx (assuming fx = fy)
    # points_2d = (image_points - camera_intrinsics[:2, 2]) / camera_intrinsics[0, 0]

    # Convert image points to homogeneous coordinates
    points_2d = np.concatenate([image_points, np.ones((len(image_points), 1))], axis=1)

    # Express points in camera coordinates
    points_3d_cam = camera_intrinsics @ points_2d.T  # points_2d.T is 3xN, z=1?
    unit_ray_cam = points_3d_cam / np.linalg.norm(points_3d_cam, axis=0)  # Nx3

    # Compute unit ray in world coordinates
    unit_ray_world = (camera_extrinsics[frame_num].T @ unit_ray_cam).T  # Nx3
    unit_ray_world = unit_ray_world / np.linalg.norm(unit_ray_world, axis=0)  # Nx3

    # Compute intersection of unit ray in world coordinates with mesh
    # Use trimesh to compute intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[camera_extrinsics[frame_num][:3, 3]],
        ray_directions=[unit_ray_world]
    )

    if len(locations) > 0:
        points_3d_world = locations[0]  # Closest intersection
        # depth = np.linalg.norm(intersection_point - camera_center)
    else:
        print('No intersection found')
        points_3d_world = None

    # Transform from camera to world coordinates
    # points_3d_world = (points_3d_cam.T @ camera_extrinsics[frame_num])[:, :3]  # Nx3

    return points_3d_world
