# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import xarray as xr
from movement.io import load_poses

# from skimage.transform import warp
# import scipy
# from scipy.spatial.transform import Rotation as R
# from movement.utils.broadcasting import make_broadcastable


%matplotlib widget


# For matrix multiplication approach:
# - try xarray broadcasting -- but what are the dimensions of the matrices? space, space?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# file_path = "/Users/sofia/swc/project_zebras/zebras-stitching/data/270Predictions_clean.analysis.h5"
file_path = "/Users/sofia/swc/project_zebras/zebras-stitching/data/20250325_2228_id.slp"
transforms_file = "/Users/sofia/swc/project_zebras/zebras-stitching/stitching-elastix/out_euler_frame.csv"
video_file = "/Users/sofia/swc/project_zebras/videos/21Jan_007.mp4"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video
video = sio.load_video(video_file)
print(video.shape)

n_frames = video.shape[0]
frame_shape = video.shape[1:]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read transforms from elastix (Euler aka 2D rot+translation only)
# transforms are expressed:
# - as a rotation around the center of the image
# - and a translation vector
# The rotation is expressed as an angle in radians
# The translation is expressed as a vector in pixels
# The transforms are from frame f to frame f+1 (ok?)

transforms_df = pd.read_csv(transforms_file)

# Add row to dataframe with transform for first frame
transforms_df = pd.concat(
    [pd.DataFrame({"theta": 0, "tx": 0, "ty": 0}, index=[0]), transforms_df],
    ignore_index=True,
)

print(f"Number of transforms: {transforms_df.shape}")
print(f"Number of frames: {n_frames}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read trajectory as a movement dataset

ds = load_poses.from_file(file_path, source_software="SLEAP")

position_array = ds.position

# %%%%%%%%%%%%%%%%%%%%%
# 1- [x] Compute homogeneous position array
# 2- [x] Transform to ICS-centre
# 3- Apply translation + rotation
# 4- Transform back to ICS corner

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute position array in homogeneous coordinates
# (x, y, 1) instead of (x, y)

homog_space = ["x", "y", "h"]
position_array_homogeneous = xr.concat(
    [
        ds["position"], 
        xr.full_like(ds["position"].sel(space="x"), 1).expand_dims(space=["h"])
    ],
    dim="space"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute change of basis matrix:

# when applied from the left to a (column) vector,
# it transforms its coordinates to express them in
# a parallel coordinate system to the ICS but with
# its origin in the image centre
v_corner_to_centre = np.array(tuple(s // 2 for s in frame_shape[:2]))
Q_corner_to_centre = np.eye(3)
Q_corner_to_centre[:2, 2] = -v_corner_to_centre

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Apply change of basis to the position array

position_homog_ICS_centre = xr.apply_ufunc(
    lambda vec: Q_corner_to_centre @ vec,  
    position_array_homogeneous,  # position in ICS corner
    input_core_dims=[["space"]],
    output_core_dims=[["space"]],
    vectorize=True
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%&&&&&&&&&&&&&
# Compute array of (accum) rotation matrices --- ok?  
# check if rotm or rotm.T? the end of the trajectory varies

def compute_rotation_matrix(theta):
    return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

f = np.vectorize(compute_rotation_matrix, signature="()->(3, 3)")

rotation_to_ICS0_array = f(
    transforms_df["theta"].cumsum().values,  # take cumulative sum of theta! it is the same right?
)

print(rotation_to_ICS0_array.shape)

#----- with xarray
# def compute_rotation_matrix(theta):
#     return R.from_matrix(
#         np.array(
#             [
#                 [np.cos(theta), -np.sin(theta), 0],
#                 [np.sin(theta), np.cos(theta), 0],
#                 [0, 0, 1],
#             ]
#         )
#     )

# rotation_to_ICS0_array = xr.apply_ufunc(
#     compute_rotation_matrix,
#     xr.DataArray(
#         transforms_df["theta"].cumsum().values,  # take cumulative sum of theta! is it the same?
#         dims=["time"]
#     ),
#     vectorize=True,
# )

# print(rotation_to_ICS0_array.shape)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute array of accumulated translation 

def compute_translation_matrix(tx, ty):
    return np.array(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )

f = np.vectorize(compute_translation_matrix, signature="(),()->(3, 3)")
translation_to_ICS0_array = f(
    transforms_df["tx"].cumsum().values, 
    transforms_df["ty"].cumsum().values
)

print(translation_to_ICS0_array.shape)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Apply transform to position_array_homogeneous.values

position_array_ICS0 =(
    np.linalg.inv(Q_corner_to_centre)  # (3, 3)
    @ rotation_to_ICS0_array  # (6294, 3, 3)
    @ translation_to_ICS0_array  # (6294, 3, 3)
    @ Q_corner_to_centre  # (3, 3)
    @ np.expand_dims(
        np.moveaxis(position_array_homogeneous.values, [0,1], [2,3]),
        axis=-1
    ) # (2, 322, 6294, 3, 1)
)

# undo reordering dimensions for broadcasting
position_array_ICS0 = np.moveaxis(position_array_ICS0, [0,1], [-2,-1]).squeeze()

# format as xarray
position_array_ICS0 = xr.DataArray(
    position_array_ICS0,
    dims=["time", "space", "keypoints", "individuals"],
    coords={
        "time": ds["time"],
        "space": ["x", "y", "h"],
        "keypoints": ds["keypoints"],
        "individuals": ds["individuals"],
    },
)

# plot
fig, ax = plt.subplots()
for ind in position_array_ICS0.individuals:
    ax.scatter(
        position_array_ICS0.mean("keypoints").sel(space="x", individuals=ind).values.flatten(),
        position_array_ICS0.mean("keypoints").sel(space="y", individuals=ind).values.flatten(),
        s=1,
        cmap="tab20",
    )
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")


# %%
# plot keypoints for one individual
ind = "track_0"
fig, ax = plt.subplots()
for f in range(position_array_ICS0.shape[0]):
    ax.plot(
        [
            position_array_ICS0.sel(time=f, individuals=ind, keypoints="H").sel(space="x"),
            position_array_ICS0.sel(time=f, individuals=ind, keypoints="T").sel(space="x")
        ],
        [
            position_array_ICS0.sel(time=f, individuals=ind, keypoints="H").sel(space="y"),
            position_array_ICS0.sel(time=f, individuals=ind, keypoints="T").sel(space="y")
        ],
        "go-",
    )

ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# @make_broadcastable()
# def apply_rotation(rotation, position_homog):
#     return rotation.apply(position_homog)

# position_homog_ICS_centre_rot = apply_rotation(
#     rotation_to_ICS_f_to_fplus1,
#     position_homog_ICS_centre,
#     broadcast_dimension=['time']
# )




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# With broadcasting:

# @make_broadcastable()
# def apply_transform_to_position_ICS_centre(position_homog_ICS_centre, accum_rotm, accum_trans_arr):
#     return accum_rotm @ (
#             accum_trans_arr @ position_homog_ICS_centre
#         )


# position_homog_ICS0_centre = apply_transform_to_position_ICS_centre(
#     position_homog_ICS_centre,
#     accum_rotm, # is it possible?
#     accum_trans_arr,
# )

# position_homog_ICS0 = transform_to_ICS_corner(position_homog_ICS0_centre, Q_corner_to_centre)


# %%%%%%%%%%%%%%%%%%%%%%
# # Define function to broadcast

# def transform_to_ICS0(
#     position_homog_ICS, 
#     change_of_basis_to_centre, 
#     accum_rot_arr_per_frame, 
#     accum_trans_arr_per_frame
# ):

#     # apply change of basis to the position
#     position_homog_ICS_centre = change_of_basis_to_centre @ position_homog_ICS

#     # apply rotation and translation to express in ICS0_centre
#     # order? we assume translation first
#     position_homog_ICS0_centre = accum_rot_arr_per_frame @ (
#         accum_trans_arr_per_frame @ position_homog_ICS_centre
#     )

#     # express the result back in ICS0
#     # (i.e. coord system with origin in the top-left centre)
#     position_homog_ICS0 = (
#         np.linalg.inv(change_of_basis_to_centre) @ position_homog_ICS0_centre
#     )

#     return position_homog_ICS0



# %%
# # compute keypoints in ECS (translated and rotated)
# position_ego_3d = xr.apply_ufunc(
#     lambda rot, trans, vec: rot.apply(vec - trans),
#     rotation2egocentric,  # rot
#     centroid_3d,  # trans
#     position_3d,  # vec
#     input_core_dims=[[], ["space"], ["space"]],
#     output_core_dims=[["space"]],
#     vectorize=True,
# )
# %%

# new_position_data = np.pad(ds["position"].values, ((0, 0), (0, 1), (0, 0), (0, 0)), constant_values=1)
# # Create a new DataArray with the expanded space dimension
# new_position = xr.DataArray(
#     new_position_data,
#     dims=["time", "space", "keypoints", "individuals"],
#     coords={
#         "time": ds["time"],
#         "space": new_space,
#         "keypoints": ds["keypoints"],
#         "individuals": ds["individuals"],
#     },
# )
