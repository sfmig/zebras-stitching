# %%
from pathlib import Path

import cv2
import numpy as np
import re
import pandas as pd
import sleap_io as sio
import tifffile

from movement.io import load_poses
from utils import load_video_to_numpy, run_registration
from utils_rotate import transform_image
from tqdm import tqdm

# %%

# Load video into a numpy array with shape (num_frames, height, width)

video_path = "21Jan_007.mp4"

# Lazy load video data using sio
video_data = sio.load_video(video_path, plugin="opencv")

# Non-lazy load video data using cv2
# video_data = load_video_to_numpy(video_path, frame_size=(1080, 1920))

# %%
# Compute masks
ds = load_poses.from_sleap_file("data/20250325_2228_id.slp", fps=30)

position_numpy = ds.position.to_numpy()
num_animals = position_numpy.shape[-1]

max_x = np.max(position_numpy[:, 0, :, :], axis=1).round().astype(int)
max_y = np.max(position_numpy[:, 1, :, :], axis=1).round().astype(int)
min_x = np.min(position_numpy[:, 0, :, :], axis=1).round().astype(int)
min_y = np.min(position_numpy[:, 1, :, :], axis=1).round().astype(int)
min_y.shape, max_y.shape, min_x.shape, max_x.shape

# %%
param_path = Path("./data/elastix/registration_params.txt")
output_file = Path("out_test.csv")
buffer = 10

with open(output_file, "w") as f:
    f.write("theta,tx,ty\n")

fused_image = np.array(cv2.cvtColor(video_data[0], cv2.COLOR_BGR2GRAY))
initial_offset = np.array([0, 0])
step = 16
threshold = 5

for i in tqdm(range(step, video_data.shape[0], step)):
    # Track to avoid mixing up the frames
    fixed_index = i - step
    moving_index = i

    fixed = fused_image
    moving = video_data[moving_index]

    # Convert to grayscale
    # fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    # fixed_gray, _ = transform_image(fixed_gray, offset=initial_offset)
    fixed_gray = fixed
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    moving_gray, offset = transform_image(moving_gray, offset=initial_offset)
    if np.any(offset != 0):
        new_moving_shape = np.array(moving_gray.shape) + offset
        new_moving = np.zeros(new_moving_shape, dtype=moving_gray.dtype)
        new_moving[:moving_gray.shape[0], :moving_gray.shape[1]] = moving_gray
        moving_gray = new_moving

    # moving_gray = moving
    fixed_mask = np.ones_like(fixed_gray, dtype=np.uint8)
    moving_mask = np.ones_like(moving_gray, dtype=np.uint8)

    fixed_mask[fixed == 0] = 0
    moving_mask[moving_gray == 0] = 0

    # for j in range(num_animals):
    #     # After casting to int, nan values become int64 min value
    #     # if (
    #     #     np.any(min_y[fixed_index, j] < 0)
    #     #     or np.any(min_x[fixed_index, j] < 0)
    #     #     or np.any(max_y[fixed_index, j] < 0)
    #     #     or np.any(max_x[fixed_index, j] < 0)
    #     # ):
    #     #     continue
    # #
    #     if (
    #         np.any(min_y[moving_index, j] < 0)
    #         or np.any(min_x[moving_index, j] < 0)
    #         or np.any(max_y[moving_index, j] < 0)
    #         or np.any(max_x[moving_index, j] < 0)
    #     ):
    #         continue
    # #
    #     # Mask out the area around the animal
    #     # fixed_mask[
    #     #     min_y[fixed_index, j] - buffer : max_y[fixed_index, j] + buffer,
    #     #     min_x[fixed_index, j] - buffer : max_x[fixed_index, j] + buffer,
    #     # ] = 0
    #     moving_mask[
    #         min_y[moving_index, j] - buffer : max_y[moving_index, j] + buffer,
    #         min_x[moving_index, j] - buffer : max_x[moving_index, j] + buffer,
    #     ] = 0

    parameters = run_registration(
        moving_image=moving_gray,
        fixed_image=fixed_gray,
        registration_parameter_path=param_path,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
    )

    # Regular expression to find the TransformParameters line
    pattern = r"\(TransformParameters ([\d\.\-e ]+)\)"
    input_string = str(parameters)
    # Search for the pattern in the input string
    match = re.search(pattern, input_string)
    if match:
        # Extract the numbers and convert them to floats
        transform_parameters = list(map(float, match.group(1).split()))
        with open(output_file, "a") as f:
            f.write(",".join(map(str, transform_parameters)))
            f.write("\n")
        # transform_matrix = np.vstack([np.array(transform_parameters).reshape((3,2)).T, [0, 0, 1]])
        # transformed_frame, orig_offset = transform_image(moving_gray, transform_matrix)
        offset = -int(transform_parameters[2]), -int(transform_parameters[1])
        # print(f"\n{transform_parameters}")
        transformed_frame, orig_offset = transform_image(moving_gray, theta=transform_parameters[0], offset=offset)
        # print(fixed_gray.shape, transformed_frame.shape, orig_offset)
        fixed_gray_shape = np.array(fixed_gray.shape) + orig_offset
        new_image_shape = np.max(np.array([fixed_gray_shape, transformed_frame.shape]), axis=0)
        # print(new_image_shape)
        fused_image = np.zeros(new_image_shape, dtype=fused_image.dtype)
        fused_image[:transformed_frame.shape[0], :transformed_frame.shape[1]] = transformed_frame
        fused_image[orig_offset[0]:orig_offset[0] + fixed_gray.shape[0], orig_offset[1]:orig_offset[1] + fixed_gray.shape[1]][
            fused_image[orig_offset[0]:orig_offset[0] + fixed_gray.shape[0], orig_offset[1]:orig_offset[1] + fixed_gray.shape[1]] < threshold
        ] = fixed_gray[fused_image[orig_offset[0]:orig_offset[0] + fixed_gray.shape[0], orig_offset[1]:orig_offset[1] + fixed_gray.shape[1]] < threshold]

        initial_offset = initial_offset + offset
        # fused_image[orig_offset[0]:orig_offset[0] + fixed_gray.shape[0], orig_offset[1]:orig_offset[1] + fixed_gray.shape[1]] = fixed_gray
        # fused_image[:transformed_frame.shape[0], :transformed_frame.shape[1]][
        #     fused_image[:transformed_frame.shape[0], :transformed_frame.shape[1]] < threshold] = transformed_frame[
        #     fused_image[:transformed_frame.shape[0], :transformed_frame.shape[1]] < threshold]
    else:
        print("TransformParameters not found")

tifffile.imwrite("test.tiff", fused_image)
# %%
transform_df = pd.read_csv(output_file)

# Add row to dataframe with transform for first frame
# For f=0, the transform from the current to the previous frame
# is the identity (i.e., no rotation, no translation)
transform_df = pd.concat(
    [pd.DataFrame({"theta": [0], "tx": [0], "ty": [0]}), transform_df],
    ignore_index=True,
)
transform_df.head()

# %%
transform_df["tx_sum"] = transform_df["tx"].cumsum().round(0).astype(int)
transform_df["ty_sum"] = transform_df["ty"].cumsum().round(0).astype(int)

x_min = transform_df["tx_sum"].min()
x_max = transform_df["tx_sum"].max()
y_min = transform_df["ty_sum"].min()
y_max = transform_df["ty_sum"].max()

print(x_min, x_max, y_min, y_max)
# %%
height, width = video_data.shape[1:3]

total_height = y_max - y_min + height
total_width = x_max - x_min + width

print(total_height, total_width)
# %%
fused_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
# Frame stride can be increased to speed up the fusing process (might be needed if video is not in RAM)
frame_stride = 1

# Naively fuse the images by placing them in the correct position
# in reverse order every nth frame
for i in tqdm(range(video_data.shape[0] - 1, 0, -frame_stride)):
    x = transform_df["tx_sum"][i] - x_min
    y = transform_df["ty_sum"][i] - y_min
    fused_image[y : y + height, x : x + width] = video_data[i]

if frame_stride > 1:
    # Ensure the 0th frame is also included
    x = transform_df["tx_sum"][0] - x_min
    y = transform_df["ty_sum"][0] - y_min
    fused_image[y : y + height, x : x + width] = video_data[0]

# Save the image to a file
tifffile.imwrite("fused_image.tif", fused_image)
