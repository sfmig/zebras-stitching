import itertools
from typing import Tuple, Optional
import numpy.typing as npt

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
import scipy.ndimage as ndimage
from PIL.Image import Image


def transform_image(image: npt.NDArray, matrix: Optional[npt.NDArray] = None, theta: float = 0.0, offset: Tuple[float, float] = (0, 0)) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Transform the image by rotating it by theta degrees and translating it by offset.

    Parameters
    ----------
    image : npt.NDArray
        The image to be transformed.
    matrix : npt.NDArray
        The transformation matrix, by default None. If provided, this matrix
        will be used to transform the image instead of the rotation and
        translation parameters.
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
    return_matrix = np.eye(3)
    return_matrix[:-1, -1] = center
    if matrix is not None:
        transform_matrix = matrix
    else:
        transform_matrix = ski.transform.EuclideanTransform(rotation=np.deg2rad(theta), translation=offset)

    shape_transform_matrix = return_matrix @ transform_matrix @ translate_matrix

    out_shape, orig_offset = calculate_transformed_bounding_box(image.shape, shape_transform_matrix)

    if np.any(orig_offset > 0):
        out_center = np.array(out_shape) // 2
        out_translate_matrix = np.eye(3)
        out_translate_matrix[:-1, -1] = center + orig_offset
    else:
        out_translate_matrix = np.eye(3)
        out_translate_matrix[:-1, -1] = center

    final_transform_matrix = out_translate_matrix @ transform_matrix @ translate_matrix

    transformed_image = ndimage.affine_transform(image, np.linalg.inv(final_transform_matrix), output_shape=out_shape, order=0)

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


def run_orb(fixed: npt.NDArray, moving: npt.NDArray) -> npt.NDArray:
    orb = ski.feature.ORB(n_keypoints=800)

    orb.detect_and_extract(fixed)
    fixed_kp, fixed_des = orb.keypoints, orb.descriptors
    orb.detect_and_extract(moving)
    moving_kp, moving_des = orb.keypoints, orb.descriptors

    # Match descriptors.
    matches = ski.feature.match_descriptors(fixed_des, moving_des, cross_check=True)

    return matches, fixed_kp, moving_kp


if __name__ == "__main__":
    from PIL import Image
    from movement.io import load_poses
    import skimage as ski
    import napari
    import pandas as pd
    import sleap_io as sio

    # Example usage
    example_image = ski.color.rgb2gray(ski.data.astronaut())

    video_data = sio.load_video("21Jan_007.mp4", plugin="opencv", grayscale=True)

    frame_zero = np.squeeze(video_data[0]).astype(np.uint8)
    frame_one = np.squeeze(video_data[16]).astype(np.uint8)
    frame_two = np.squeeze(video_data[32]).astype(np.uint8)

    matches, fixed_kp, moving_kp = run_orb(frame_zero, frame_two)
    print(matches.shape)

    ds = load_poses.from_sleap_file("data/20250325_2228_id.slp", fps=30)

    position_numpy = ds.position.to_numpy()
    num_animals = position_numpy.shape[-1]

    max_x = np.max(position_numpy[:, 0, :, :], axis=1).round().astype(int)
    max_y = np.max(position_numpy[:, 1, :, :], axis=1).round().astype(int)
    min_x = np.min(position_numpy[:, 0, :, :], axis=1).round().astype(int)
    min_y = np.min(position_numpy[:, 1, :, :], axis=1).round().astype(int)
    min_y.shape, max_y.shape, min_x.shape, max_x.shape

    fixed_index = 0
    fixed_mask = np.ones_like(frame_zero, dtype=np.bool)
    buffer = 5

    for j in range(num_animals):
        # After casting to int, nan values become int64 min value
        if (
                np.any(min_y[fixed_index, j] < 0)
                or np.any(min_x[fixed_index, j] < 0)
                or np.any(max_y[fixed_index, j] < 0)
                or np.any(max_x[fixed_index, j] < 0)
        ):
            continue

        # if (
        #     np.any(min_y[moving_index, j] < 0)
        #     or np.any(min_x[moving_index, j] < 0)
        #     or np.any(max_y[moving_index, j] < 0)
        #     or np.any(max_x[moving_index, j] < 0)
        # ):
        #     continue
        #
        # Mask out the area around the animal
        min_y_adj, max_y_adj = min_y[fixed_index, j], max_y[fixed_index, j]
        min_x_adj, max_x_adj = min_x[fixed_index, j], max_x[fixed_index, j]
        fixed_mask[
        min_y_adj - buffer: max_y_adj + buffer,
        min_x_adj - buffer: max_x_adj + buffer,
        ] = 0

    clean_matches = np.array([match for match in matches if fixed_mask[int(fixed_kp[match[0]][0]), int(fixed_kp[match[0]][1]) ] > 0])
    print(clean_matches.shape)

    src_points = np.array([fixed_kp[match[0]] for match in clean_matches])
    dst_points = np.array([moving_kp[match[1]] for match in clean_matches])

    transform = ski.transform.EuclideanTransform()
    if transform.estimate(src_points, dst_points):
        print(transform.translation, transform.rotation)

    new_shape = (frame_zero.shape[0] + 100, frame_zero.shape[1] + 100)
    fused_image = np.zeros(new_shape)
    fused_image[:frame_zero.shape[0], :frame_zero.shape[1]] = frame_zero
    translation = transform.inverse.translation
    fused_image[int(translation[0]):int(translation[0]) + frame_one.shape[0], int(translation[1]):int(translation[1]) + frame_one.shape[1]] = frame_one

    plt.imshow(fused_image)
    plt.show()
    # transformed_image, orig_offset = transform_image(frame_one, theta=0, offset=(7, 22))
    #
    # print(transformed_image.shape, orig_offset)
    #
    # h, w = frame_zero.shape
    # #
    # # output_file = "data/elastix/out_euler_frame_masked.csv"
    # # transform_df = pd.read_csv(output_file)
    # #
    # # # Add row to dataframe with transform for first frame
    # # # For f=0, the transform from the current to the previous frame
    # # # is the identity (i.e., no rotation, no translation)
    # # transform_df = pd.concat(
    # #     [pd.DataFrame({"theta": [0], "tx": [0], "ty": [0]}), transform_df],
    # #     ignore_index=True,
    # # )
    # #
    # # transformed_frame, orig_offset = transform_image(frame_one, theta=transform_df["theta"][1], offset=(transform_df["tx"][1], transform_df["ty"][1]))
    # print(orig_offset[0] + h - orig_offset[0])
    # new_image_shape = np.max([transformed_image.shape, np.array(frame_zero.shape) + orig_offset], axis=0)
    # fused_image = np.zeros(new_image_shape, dtype=np.uint8)
    # fused_image[orig_offset[0]:orig_offset[0] + h, orig_offset[1]:orig_offset[1] + w] = frame_zero
    # # fused_image[0:transformed_image.shape[0], 0:transformed_image.shape[1]] = transformed_image
    # fused_image[:transformed_image.shape[0], :transformed_image.shape[1]][fused_image[:transformed_image.shape[0], :transformed_image.shape[1]] == 0] = transformed_image[fused_image[:transformed_image.shape[0], :transformed_image.shape[1]] == 0]
    # #
    # # # Display the transformed frame
    # viewer = napari.Viewer()
    #
    # viewer.add_image(frame_zero, name="Frame zero")
    # viewer.add_image(frame_one, name="Frame one")
    # viewer.add_image(transformed_image, name="Transformed frame one")
    # viewer.add_image(fused_image, name="Fused Image")
    # #
    # # if np.any(orig_offset > 0):
    # #     new_shape = example_image.shape + orig_offset
    # #     transformed_image = np.zeros(new_shape)
    # #     transformed_image[orig_offset[0]:orig_offset[0] + h, orig_offset[1]:orig_offset[1] + w] = example_image
    # #     viewer.add_image(transformed_image, name="Original Image with Offset")
    # #
    # viewer.show(block=True)