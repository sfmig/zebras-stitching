from typing import Tuple
import numpy.typing as npt

import cv2
import numpy as np
import skimage as ski


def transform_image(image: npt.NDArray, theta: float = 0.0, offset: Tuple[int, int] = (0, 0)) -> Tuple[npt.NDArray, npt.NDArray]:
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
    Tuple[npt.NDArray, npt.NDArray]
        The transformed image and the origin offset.
    """
    center = np.array(image.shape) // 2

    translate_matrix = np.eye(3)
    translate_matrix[:-1, -1] = -center
    transform_matrix = ski.transform.EuclideanTransform(rotation=np.deg2rad(theta), translation=offset)

    shape_transform_matrix = np.linalg.inv(translate_matrix) @ transform_matrix @ translate_matrix

    out_shape, orig_offset = calculate_transformed_bounding_box(image.shape, shape_transform_matrix)

    if np.any(orig_offset > 0):
        out_center = np.array(out_shape) // 2
        out_translate_matrix = np.eye(3)
        out_translate_matrix[:-1, -1] = center + orig_offset
    else:
        out_translate_matrix = np.eye(3)
        out_translate_matrix[:-1, -1] = center

    final_transform_matrix = out_translate_matrix @ transform_matrix @ translate_matrix

    transformed_image = cv2.warpAffine(image, final_transform_matrix[:-1, :], out_shape[::-1])

    return transformed_image, orig_offset


def calculate_transformed_bounding_box(
    image_shape: Tuple[int, int], rotation_matrix: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the bounding box of the rotated image.

    Calculates the bounding box of the rotated image given the
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
    npt.NDArray
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

    output_shape = np.where(min_corner < 0, np.ceil(max_corner - min_corner).astype(np.int32), np.ceil(max_corner).astype(np.int32))[:-1]
    origin_offset = np.where(min_corner < 0, np.ceil(np.abs(min_corner).astype(np.int32)), 0)[:-1]

    return output_shape, origin_offset


if __name__ == "__main__":
    import skimage as ski
    import napari
    import pandas as pd
    import sleap_io as sio

    # Example usage
    example_image = ski.color.rgb2gray(ski.data.astronaut())
    h, w = example_image.shape

    video_data = sio.load_video("21Jan_007.mp4", plugin="opencv", grayscale=True)

    frame_zero = np.squeeze(video_data[0])
    frame_one = np.squeeze(video_data[1])

    h, w = frame_zero.shape

    output_file = "data/elastix/out_euler_frame_masked.csv"
    transform_df = pd.read_csv(output_file)

    # Add row to dataframe with transform for first frame
    # For f=0, the transform from the current to the previous frame
    # is the identity (i.e., no rotation, no translation)
    transform_df = pd.concat(
        [pd.DataFrame({"theta": [0], "tx": [0], "ty": [0]}), transform_df],
        ignore_index=True,
    )

    transformed_frame, orig_offset = transform_image(frame_one, theta=transform_df["theta"][1], offset=(transform_df["tx"][1], transform_df["ty"][1]))
    fused_image = np.copy(transformed_frame)
    fused_image[orig_offset[0]:orig_offset[0] + h, orig_offset[1]:orig_offset[1] + w] = frame_zero
    # transformed_frame, orig_offset = transform_image(example_image, theta=45, offset=(200, 200))

    # Display the transformed frame
    viewer = napari.Viewer()

    viewer.add_image(frame_zero, name="Original Image")
    viewer.add_image(transformed_frame, name="Transformed Image")
    viewer.add_image(fused_image, name="Fused Image")

    if np.any(orig_offset > 0):
        new_shape = example_image.shape + orig_offset
        transformed_image = np.zeros(new_shape)
        transformed_image[orig_offset[0]:orig_offset[0] + h, orig_offset[1]:orig_offset[1] + w] = example_image
        viewer.add_image(transformed_image, name="Original Image with Offset")

    viewer.show(block=True)