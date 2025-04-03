from typing import List, Optional, Tuple
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

    elastix_object = itk.ElastixRegistrationMethod.New(
        fixed_image, moving_image
    )

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