# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import xarray as xr
from movement.io import load_poses
from skimage.transform import warp

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# file_path = "/Users/sofia/swc/project_zebras/zebras-stitching/data/270Predictions_clean.analysis.h5"
file_path = "/Users/sofia/swc/project_zebras/zebras-stitching/data/Annotators - merged.slp.250323_203032.predictions.slp"
transforms_file = "/Users/sofia/swc/project_zebras/zebras-stitching/stitching-elastix/out_euler_frame.csv"
video_file = "/Users/sofia/swc/project_zebras/videos/21Jan_007.mp4"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video
video = sio.load_video(video_file)
print(video.shape)

n_frames = video.shape[0]
frame_shape = video.shape[1:]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read trajectory as a movement dataset

ds = load_poses.from_file(file_path, source_software="SLEAP")

# Reduce to mean of keypoints
# ds = ds.mean("keypoints")

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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get coordinates of selected tracklet in ICS
# a tracklet is a (correct) trajectory of one individual

frame_start = 0  # 1467
# frame_end = 6293  # 1975

# ICS = image coordinate system
tracklet_position_ICS = ds.position  # .sel(time=slice(frame_start, frame_end, 1))
tracklet_centroid_ICS = tracklet_position_ICS.squeeze().mean("keypoints")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot tracklet in ICS
fig, ax = plt.subplots()
im = ax.imshow(video[frame_start])

for ind in tracklet_centroid_ICS.individuals:
    sc = ax.scatter(
        tracklet_centroid_ICS.sel(space="x", individuals=ind),
        tracklet_centroid_ICS.sel(space="y", individuals=ind),
        c=tracklet_centroid_ICS.time,
        cmap="viridis",
        s=10,
    )
# ax.set_xlim(875, 1075)
# ax.set_ylim(660, 876)
ax.set_aspect("equal")
ax.invert_yaxis()

ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

cbar = plt.colorbar(sc)
cbar.set_label("frame")

ax.set_title("Centroid - ICS")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read transforms for tracklet
# ATT! first row is the transform from f=1466 to f=1467!
transforms_tracklet = transforms_df  # .iloc[slice(frame_start, frame_end + 1, 1)]
print(transforms_tracklet)

# check as many transforms as frames in tracklet
print(tracklet_centroid_ICS.shape)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot accumulated rotated angle
plt.figure()
plt.plot(np.rad2deg(transforms_tracklet["theta"].cumsum()), "-o")
plt.title("Rotation wrt ICS0-x-axis")
plt.xlabel("frame")
plt.ylabel("theta (deg)")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extract transform per frame in homogeneous coords
# transforms are from frame f --> f+1

translation_arrays_per_frame = [
    np.vstack((np.hstack((np.eye(2), t.reshape(-1, 1))), np.array([0, 0, 1])))
    for t in transforms_tracklet[["tx", "ty"]].values
]

# replace transform from f=1466 to f=1467 with null translation
# translation_arrays_per_frame[0] = np.eye(3)

# rotation matrices from f to f+1
# with rotation axis = center of the image
rot_matrices_per_frame = [
    np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    for theta in transforms_tracklet["theta"].values
]

# replace rotation matrix from f=1466 to f=1467 with identity
rot_matrices_per_frame[0] = np.eye(3)

print(len(translation_arrays_per_frame))
print(len(rot_matrices_per_frame))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute transforms from reference frame f=0 in homog coords

# translation
accum_translation_arrays_per_frame = []

accum_trans_arr = np.eye(3)
for trans_arr in translation_arrays_per_frame:
    accum_trans_arr = trans_arr @ accum_trans_arr
    accum_translation_arrays_per_frame.append(accum_trans_arr)


# rotation
# multiply from the left
accum_rot_matrices_per_frame = []
accum_rot_arr = np.eye(3)
for rot_arr in rot_matrices_per_frame:
    accum_rot_arr = rot_arr @ accum_rot_arr
    accum_rot_matrices_per_frame.append(accum_rot_arr)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute change of basis matrix:

# when applied from the left to a (column) vector,
# it transforms its coordinates to express them in
# a parallel coordinate system to the ICS but with
# its origin in the image centre
vector_to_img_centre = np.array(tuple(s // 2 for s in frame_shape[:2]))
change_of_basis_to_centre = np.vstack(
    (
        np.hstack(
            [
                np.eye(2),
                np.array(-vector_to_img_centre).reshape(-1, 1),
            ]
        ),
        np.array([0, 0, 1]),
    )
)

# Note: a change of basis matrix is the "inverse" of the
# corresponding rotation. So the application of this matrix
# to a point can also be seen as translating the data such that
# the image centre is at the origin of our ICS.

# check the top-left corner in ICS is negative x and negative y in
# ICS-centre
# (we use homogeneous coords)
print(change_of_basis_to_centre @ np.array([0, 0, 1]))

# check the inverse transforms points in ICS-centre to points expressed
# in ICS
print(np.linalg.inv(change_of_basis_to_centre) @ np.array([0, 0, 1]))

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute tracklet coordinates in ICSO

# accum_trans_arr = np.eye(3)

# For each individual
list_tracklet_centroid_ICS0_all_individuals = []
for id in range(tracklet_centroid_ICS.values.shape[2]):
    list_tracklet_centroid_ICS0 = []

    # For each frame
    for accum_trans_arr, accum_rot_arr, position_s0 in zip(
        accum_translation_arrays_per_frame,
        accum_rot_matrices_per_frame,
        tracklet_centroid_ICS.values[:, :, id],
        strict=True,
    ):
        # get position data in homog coords in ICS of current frame
        position_s0_homog = np.vstack([position_s0.reshape(-1, 1), 1.0])

        # express position data in ICS-centre of current frame:
        # coord system with origin in the center of the image
        position_s0_homog_centre = change_of_basis_to_centre @ position_s0_homog

        # apply rotation and translation in ICS-centre
        # order? we assume translation first
        position_s1_homog_centre = accum_rot_arr @ (
            accum_trans_arr @ position_s0_homog_centre
        )

        # express the result back in ICS
        # (i.e. coord system with origin in the top-left centre)
        position_s1_homog = (
            np.linalg.inv(change_of_basis_to_centre) @ position_s1_homog_centre
        )

        list_tracklet_centroid_ICS0.append(position_s1_homog[:-1, :].T.squeeze())

    list_tracklet_centroid_ICS0_all_individuals.append(
        np.array(list_tracklet_centroid_ICS0)
    )

# format as xarray
tracklet_centroid_ICS0 = xr.DataArray(
    np.stack(list_tracklet_centroid_ICS0_all_individuals, axis=-1),
    dims=["time", "space", "individuals"],
    coords={
        "time": range(frame_start, tracklet_centroid_ICS.values.shape[0]),
        "space": ["x", "y"],
    },
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute blended image using one very n frames
blend_step = 50
list_frames = list(range(frame_start, tracklet_centroid_ICS.values.shape[0]))
list_frames_to_plot = list_frames[0:-1:blend_step]

# output_shape = [int(x * 5) for x in frame_shape[:2]]
output_shape = [1400, 5500]


blended_warped_img = np.zeros(output_shape + [3])
for f_i, f in enumerate(list_frames_to_plot):
    img_warped = warp(
        video[f],
        np.linalg.inv(
            np.linalg.inv(change_of_basis_to_centre)
            @ accum_rot_matrices_per_frame[list_frames.index(f)]
            @ accum_translation_arrays_per_frame[list_frames.index(f)]
            @ change_of_basis_to_centre
        ),
        # we do inverse outside because skimage's warp expects
        # the transform from output image to input image
        output_shape=output_shape,
    )
    blended_warped_img = np.maximum(blended_warped_img, img_warped)

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot  trajectory in ICS0
# ax = axs
fig, ax = plt.subplots()
ax.imshow(blended_warped_img)

for ind in tracklet_centroid_ICS0.individuals:
    sc = ax.scatter(
        tracklet_centroid_ICS0.sel(space="x", individuals=ind),
        tracklet_centroid_ICS0.sel(space="y", individuals=ind),
        c=tracklet_centroid_ICS0.time,
        cmap="viridis",
        s=1,
    )
# ax.set_xlim(875, 1075)
# ax.set_ylim(660, 876)
ax.set_aspect("equal")
# ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
plt.colorbar(sc)

ax.set_title("Centroid - ICS0")

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot  trajectory in ICS0
# ax = axs
fig, ax = plt.subplots()
ax.imshow(blended_warped_img)

for t in range(tracklet_centroid_ICS0.shape[0]):
    ax.scatter(
        tracklet_centroid_ICS0[t, 0, :],
        tracklet_centroid_ICS0[t, 1, :],
        c=range(tracklet_centroid_ICS0.shape[2]),
        s=1,
        cmap="tab20",
    )
# ax.set_xlim(875, 1075)
# ax.set_ylim(660, 876)
ax.set_aspect("equal")
# ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
# plt.colorbar(sc)

ax.set_title("Centroid - ICS0")


# %%
# from skimage.transform import SimilarityTransform
# # tform = SimilarityTransform(translation=(0, -10))
# tform = SimilarityTransform(np.linalg.inv(accum_translation_arrays_per_frame[500]))
# warped = warp(
#     video[0],
#     tform,
#     output_shape=[x * 1.75 for x in frame_shape[:2]],
#     # clip=False,
# )

# fig, ax = plt.subplots()
# ax.imshow(warped)

# %%
from pathlib import Path

# Define a movement dataset
from movement.io import load_poses, save_poses

ds_export = load_poses.from_numpy(
    position_array=np.expand_dims(tracklet_centroid_ICS0.values, axis=-2),
    confidence_array=np.expand_dims(ds.confidence.mean("keypoints"), axis=-2),
    individual_names=ds.individuals.values,
    keypoint_names=["centroid"],
    source_software="manual",
)

ds_export.attrs["source_file"] = ""

slp_file = save_poses.to_sleap_analysis_file(
    ds_export, Path("/Users/sofia/swc/project_zebras/zebras-stitching/test.h5")
)

# %%
# %%
