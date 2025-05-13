"""A notebook to read the data transformed using OpenSFM poses and save it in a napari-compatible format."""

# %%
from datetime import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from movement.io import load_poses, save_poses

%matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
data_dir = Path("data")
# points_3d_npz = data_dir / "20250325_2228_id_sfm_3d_unwrapped_20250512_080214.npz"
points_3d_npz = data_dir / "20250325_2228_id_sfm_3d_unwrapped_plane_20250512_215051.npz"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read the npz file
data_dict = np.load(points_3d_npz)

# Reorder position array: 
# it follows 0.0.5 convention: (time, individuals, keypoints, space)
# I want: (time, space, keypoints, individuals)
position = data_dict["position"].transpose((0, 3, 2, 1))
print(position.shape)

# Reorder confidence array:
# it follows 0.0.5 convention: (time, individuals, keypoints)
# I want: (time, keypoints, individuals)
confidence = data_dict["confidence"].transpose((0, 2, 1))
print(confidence.shape)


print(len(data_dict["individuals"]))
print(len(data_dict["keypoints"]))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create movement dataset
ds = load_poses.from_numpy(
    position_array=position, #[:,:2,:, :], # remove z-coordinates
    confidence_array=confidence,
    individual_names=data_dict["individuals"],
    keypoint_names=data_dict["keypoints"],
    fps=None,
    source_software='sfm',
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot trajectories in 3D

fig = plt.figure()
ax = fig.add_subplot(111) #, projection='3d')
color_per_frame = plt.cm.viridis(np.linspace(0, 1, ds.time.size))
color_per_individual = plt.cm.turbo(np.linspace(0, 1, ds.individuals.size))
for individual_idx in range(ds.individuals.size):
    for frame_idx in range(ds.time.size):
        # line
        # (time, space, keypoints, individuals)
        ax.plot(
            ds.position[frame_idx, 0, :, individual_idx], 
            ds.position[frame_idx, 1, :, individual_idx], 
            # ds.position[frame_idx, 2, :, individual_idx], 
            '-',
            label=f"frame {frame_idx}",
            c=color_per_frame[frame_idx]
        )
        # # head
        # ax.scatter(
        #     ds.position[frame_idx, 0, 0, individual_idx], 
        #     ds.position[frame_idx, 1, 0, individual_idx], 
        #     ds.position[frame_idx, 2, 0, individual_idx], 
        #     'o',
        #     c=color_per_individual[individual_idx]
        # )
        # # tail
        # ax.scatter(
        #     ds.position[frame_idx, 0, 1, individual_idx], 
        #     ds.position[frame_idx, 1, 1, individual_idx], 
        #     ds.position[frame_idx, 2, 1, individual_idx], 
        #     '*',
        #     c=color_per_individual[individual_idx]
        # )
    
# plt.legend()
ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
# ax.set_zlabel('z (pixels)')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save movement dataset for napari viz

ds.attrs["source_file"] = points_3d_npz

# get string timestamp of  today in yyyymmdd_hhmmss
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Drop z-coordinate before exporting
# TODO: project to plane fitted to mesh instead
ds_2d_export = ds.drop_sel(space="z")

# Change direction of x axis
# TODO: use basis defined on plane
ds_2d_export.position.values[:, 0, :, :] *= -1

slp_file = save_poses.to_sleap_analysis_file(
    ds_2d_export,
    data_dir / f"{points_3d_npz.stem}_2d.h5",
)
# %%
