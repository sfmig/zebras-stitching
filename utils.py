from typing import Optional, Tuple
from pathlib import Path
import numpy.typing as npt

import cv2
import itk
import numpy as np


def run_registration(
    moving_image: npt.NDArray,
    fixed_image: npt.NDArray,
    registration_parameter_path: Path,
    moving_mask: Optional[npt.NDArray] = None,
    fixed_mask: Optional[npt.NDArray] = None,
) -> itk.ParameterObject:
    """
    Run the registration process on the given images.

    Parameters
    ----------
    moving_image : npt.NDArray
        The fixed image.
    fixed_image : npt.NDArray
        The moving image.
    registration_parameter_path : Path
        The path to the registration parameter file.
    moving_mask : Optional[npt.NDArray], optional
        The atlas mask, by default None. If provided, the mask will be used
        to exclude regions from the registration.
    fixed_mask : Optional[npt.NDArray], optional
        The moving mask, by default None. If provided, the mask will be used
        to exclude regions from the registration.

    Returns
    -------
    itk.ParameterObject
        The result transform parameters.
    """
    # convert to ITK, view only
    moving_image = itk.GetImageViewFromArray(moving_image)
    fixed_image = itk.GetImageViewFromArray(fixed_image)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    if moving_mask is not None:
        moving_mask = itk.GetImageViewFromArray(moving_mask)
        elastix_object.SetMovingMask(moving_mask)

    if fixed_mask is not None:
        fixed_mask = itk.GetImageViewFromArray(fixed_mask)
        elastix_object.SetFixedMask(fixed_mask)

    parameter_object = itk.ParameterObject.New()
    parameter_object.ReadParameterFile(str(registration_parameter_path))

    elastix_object.SetParameterObject(parameter_object)

    # Run the registration
    elastix_object.UpdateLargestPossibleRegion()

    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return result_transform_parameters


def load_video_to_numpy(video_path, frame_size=(1080, 1920)):
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = np.zeros((num_frames, *frame_size, 3), dtype=np.uint8)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames[frame_index] = frame
        frame_index += 1

    cap.release()

    return frames


def transform_image(image: npt.NDArray, theta: float = 0.0, offset: Tuple[int, int] = (0, 0)):
    """
    Transform the image by rotating it by theta degrees and translating it by offset.

    Parameters
    ----------
    image : npt.NDArray
        The image to be transformed.
    theta : float, optional
        The angle of rotation in degrees, by default 0.0
    offset : tuple, optional
        The translation offset (x, y), by default (0, 0)

    Returns
    -------
    npt.NDArray
        The transformed image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    M[0, 2] += offset[0]
    M[1, 2] += offset[1]

    transformed_image = cv2.warpAffine(image, M, (w+offset[0], h+offset[1]))

    return transformed_image


def calculate_transformed_bounding_box(
    image_shape: Tuple[int, int], rotation_matrix: npt.NDArray
) -> Tuple[int, int]:
    """
    Calculates the bounding box of the rotated image.

    This function calculates the bounding box of the rotated image given the
    image shape and rotation matrix. The bounding box is calculated by
    transforming the corners of the image and finding the minimum and maximum
    values of the transformed corners.

    Parameters
    ------------
    image_shape : Tuple[int, int]
        The shape of the image.
    rotation_matrix : npt.NDArray
        The rotation matrix.

    Returns
    --------
    Tuple[int, int]
        The bounding box of the rotated image.
    """
    corners = np.array(
        [
            [0, 0, 1],
            [image_shape[0], 0, 1],
            [0, image_shape[1], 1],
            [0, 0,  1],
            [image_shape[0], image_shape[1], 1],
            [image_shape[0], 0, 1],
            [0, image_shape[1], 1],
            [image_shape[0], image_shape[1], 1],
        ]
    )

    transformed_corners = np.dot(rotation_matrix, corners.T)
    min_corner = np.min(transformed_corners, axis=1)
    max_corner = np.max(transformed_corners, axis=1)

    return np.ceil(max_corner - min_corner).astype(int)


if __name__ == "__main__":
    import skimage as ski
    import napari

    # Example usage
    example_image = ski.color.rgb2gray(ski.data.astronaut())

    # Transform the first frame
    transformed_frame = transform_image(example_image, offset=(100, 100))

    # Display the transformed frame
    viewer = napari.Viewer()

    viewer.add_image(example_image, name="Original Image")
    viewer.add_image(transformed_frame, name="Transformed Image")

    viewer.show(block=True)
