
"""Compute 3d trajectories using SFM output and mesh.

Requires launching this in the opendm container.
"""
# %%
import numpy as np
import trimesh
import xarray as xr
from opensfm import dataset # requires launching this in the opendm container
from movement.io import load_poses
import matplotlib.pyplot as plt

# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
opensfm_dir = "/workspace/datasets/project/opensfm"
mesh_path = (
    "/workspace/datasets/project/odm_meshing/odm_25dmesh.ply"
)

points_2d_slp = (
    "/workspace/zebras-stitching/data/20250325_2228_id.slp"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Helper functions
def position_array_to_homogeneous(position_array: np.ndarray) -> np.ndarray:
    """
    Convert a position array to a homogeneous coordinate array.

    (x, y, 1) instead of (x, y)
    I use "h" for the third homog coord instead of "z" for clarity
    """
    return xr.concat(
        [
            position_array,
            xr.full_like(position_array.sel(space="x"), 1).expand_dims(space=["h"]),
        ],
        dim="space",
    )

def H_norm_to_pixel_coords(w, h):
    """
    Convert normalized coordinates to pixel coordinates

    https://opensfm.org/docs/geometry.html
    """
    s = max(w, h)
    return np.array([[s, 0, (w - 1) / 2], [0, s, (h - 1) / 2], [0, 0, 1]])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read mesh
mesh = trimesh.load(mesh_path)

# # If the mesh is a scene, combine all geometry into a single mesh
# if isinstance(mesh, trimesh.Scene):
#     mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read reconstruction data
data = dataset.DataSet(opensfm_dir)
recs = data.load_reconstruction()[0]



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read 2D trajectories
ds = load_poses.from_sleap_file(points_2d_slp) 

position = ds.pose_tracks  # as xarray data array, time in frames
position_homogeneous = position_array_to_homogeneous(position)
position_homogeneous_shape = position_homogeneous.shape  # time, individuals, kpts, space (homogeneous)

all_points_2d_homogeneous = position_homogeneous.values.reshape(-1,3) # as (K, 3) numpy array
assert (all_points_2d_homogeneous.reshape(position_homogeneous_shape) == position_homogeneous.values).all
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute 3D points

# Get frame filenames
list_frame_filenames = list(recs.shots.keys())
list_frame_filenames = sorted(list_frame_filenames)

# Get camera width and height
image_width = recs.shots[list_frame_filenames[0]].camera.width
image_height = recs.shots[list_frame_filenames[0]].camera.height

# Get conversion matrix from normalised to pixel coordinates
# [Note: I think all opensfm matrices return results in normalised coordinates
# But that is fine because the mesh is also in normalised coordinates]
H_pixel2norm = np.linalg.inv(H_norm_to_pixel_coords(image_width, image_height))


# %%
# Loop thru frames

list_3D_points_per_frame = []
for frame_filename in list_frame_filenames:

    # Get intrinsic and extrinsic camera parameters for this frame
    shot = recs.shots[frame_filename]
    cam = shot.camera # instrinsic matrix?
    pose = shot.pose # extrinsic

    # Get 2D pixel & **homogeneous** coordinates of points at this frame
    # (M,3 array, with M = total number of points)
    # (pixel coordinates = not normalised)
    frame_idx = int(frame_filename.split(".")[0])
    pt2D_pixels_homogeneous_shape = position_homogeneous.sel(time=frame_idx).shape  # 44,2,3
    pt2D_pixels_homogeneous = position_homogeneous.sel(time=frame_idx).values.reshape(-1,3)  


    # Compute *normalised* coordinates in camera coordinate system (aka bearings)
    pt2D_bearings_homogeneous_ccs = (
        np.linalg.inv(cam.get_K_in_pixel_coordinates(image_width, image_height)) 
        @ pt2D_pixels_homogeneous.T
    ).T  # returns normalised coordinates!
    # equivalent to:
    # pt2D_norm = (H_pixel2norm @ pt2D_pixels_homogeneous.T).T
    # bearings_homogeneous_ccs = (
    #     np.linalg.inv(cam.get_K())
    #     @ pt2D_norm.T
    # ).T  # returns normalised coordinates


    # Compute depth from intersection of ray with mesh
    # 1- compute free vector in world coordinates
    bearings_rotated_to_wcs = (pose.get_R_cam_to_world() @ pt2D_bearings_homogeneous_ccs.T).T
    bearings_rotated_to_wcs = bearings_rotated_to_wcs / np.linalg.norm(bearings_rotated_to_wcs, axis=1, keepdims=True)


    # 2- compute intersection with mesh
    # exclude nans
    pt3D_world_w_nans = np.nan * np.ones_like(pt2D_pixels_homogeneous)
    slc_nan_bearings = np.isnan(bearings_rotated_to_wcs).any(axis=1)
    pt3D_world, _, _ = mesh.ray.intersects_location(
        ray_origins=np.tile(pose.get_t_cam_to_world(), (sum(~slc_nan_bearings), 1)),  
        # pose.get_cam_to_world()[:, 3] == pose.get_t_cam_to_world()
        # pose.get_cam_to_world() @ [0,0,0,1].T == pose.get_t_cam_to_world()
        ray_directions=bearings_rotated_to_wcs[~slc_nan_bearings, :],
    )

    # Reshape to original shape
    pt3D_world_w_nans[~slc_nan_bearings, :] = pt3D_world
    pt3D_world_w_nans = pt3D_world_w_nans.reshape(pt2D_pixels_homogeneous_shape)

    # append
    list_3D_points_per_frame.append(pt3D_world_w_nans)


# Concatenate all 3D points
pt3D_world_all = np.stack(list_3D_points_per_frame, axis=0)

print(pt3D_world_all.shape)

# %%
# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(
    pt3D_world_all[:, 0, 0, 0], 
    pt3D_world_all[:, 0, 0, 1], 
    pt3D_world_all[:, 0, 0, 2], 
    '.-',
    # alpha=1,
)

ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# %%
