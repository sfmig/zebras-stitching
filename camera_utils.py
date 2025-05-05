import json
from pathlib import Path
from typing import Dict

import numpy as np


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


def get_camera_distortion(
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
    distortion = get_camera_distortion("cameras.json")

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
