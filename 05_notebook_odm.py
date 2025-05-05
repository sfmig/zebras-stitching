# %%
import numpy as np
import trimesh
import xarray as xr
from camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_camera_poses,
)
from movement.io import load_poses

# %matplotlib widget
# %%
# Input data
# mesh_path = "/home/sminano/swc/project_zebras/zebras-stitching/odm-data/odm_textured_model_geo.obj"
mesh_path = (
    "/home/sminano/swc/project_zebras/datasets/project/odm_meshing/odm_25dmesh.ply"
)

intrinsics_json = (
    "/home/sminano/swc/project_zebras/zebras-stitching/odm-data/cameras.json"
)
extrinsics_json = (
    "/home/sminano/swc/project_zebras/zebras-stitching/odm-data/shots.geojson"
)

points_2d_slp = (
    "/home/sminano/swc/project_zebras/zebras-stitching/data/20250325_2228_id.slp"
)

# %%
image_width = 1920
image_height = 1080

# %%%%%%%%%%%%%%%
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
    """
    s = max(w, h)
    return np.array([[s, 0, (w - 1) / 2], [0, s, (h - 1) / 2], [0, 0, 1]])


H_norm2pixel = H_norm_to_pixel_coords(image_width, image_height)
# %%%%%%%%%%%%%%
# Read mesh
mesh = trimesh.load(mesh_path)

# If the mesh is a scene, combine all geometry into a single mesh
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get camera intrinsics
# in normalized pixel coordinates!
camera_intrinsics = get_camera_intrinsic_matrix(intrinsics_json)

# Get camera extrinsics
camera_poses = get_camera_poses(extrinsics_json)
camera_extrinsics = get_camera_extrinsic_matrix(camera_poses)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get 2D points
ds = load_poses.from_file(points_2d_slp, source_software="SLEAP")

# Make points homogeneous
points_2d = position_array_to_homogeneous(ds.position)  # (6294, 3, 2, 44)

# Express in normalized pixel coordinates
points_2d_norm = (
    np.linalg.inv(H_norm2pixel)  # (3, 3)
    @ np.expand_dims(
        np.moveaxis(points_2d.values, [0, 1], [2, 3]), axis=-1
    )  # (2, 44, 6294, 3, 1)
    # we move the array axes to the end as per numpy.matmul convention
    # https://numpy.org/doc/2.0/reference/generated/numpy.matmul.html --> Notes
)  # (2, 44, 6294, 3, 1)


# undo the reordering dimensions required for broadcasting
points_2d_norm = np.moveaxis(
    points_2d_norm, [0, 1], [-2, -1]
).squeeze()  # (6294, 3, 2, 44)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute unit ray in camera coordinates

# Compute points in camera coordinates assuming depth=1
# The coordinates (x,y,1) represent a direction (a ray)
# from the camera center through the image plane.
points_ray_cam = (
    np.linalg.inv(camera_intrinsics)  # (3, 3)
    @ np.expand_dims(
        np.moveaxis(points_2d_norm, [0, 1], [2, 3]), axis=-1
    )  # (2, 44, 6294, 3, 1)
    # we move the array axes to the end as per numpy.matmul convention
    # https://numpy.org/doc/2.0/reference/generated/numpy.matmul.html --> Notes
)  # (2, 44, 6294, 3, 1)

# undo the reordering dimensions required for broadcasting
points_ray_cam = np.moveaxis(
    points_ray_cam, [0, 1], [-2, -1]
).squeeze()  # (6294, 3, 2, 44)

# compute unit ray in camera coordinates
# ray_cam_norm = np.linalg.norm(points_ray_cam, axis=1)  # (6294, 2, 44)
# unit_ray_cam = points_ray_cam / np.expand_dims(ray_cam_norm, axis=1)  # (6294, 3, 2, 44)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
frame_num = 1  # 1-indexed
# distortion = None

# Compute camera point ray in world coordinates
points_ray_world = (
    np.linalg.inv(camera_extrinsics[frame_num][:3, :3])  # (3, 3), rotation only
    # OJO we use camera pose
    @ np.expand_dims(
        np.moveaxis(points_ray_cam, [0, 1], [2, 3]), axis=-1
    )  # (2, 44, 6294, 3, 1)
)

# undo the reordering dimensions required for broadcasting
points_ray_world = np.moveaxis(
    points_ray_world, [0, 1], [-2, -1]
).squeeze()  # (6294, 3, 2, 44)


# %%
# normalize unit ray in world coordinates
ray_world_norm = np.linalg.norm(points_ray_world, axis=1)  # (6294, 2, 44)
unit_ray_world = points_ray_world / np.expand_dims(
    ray_world_norm, axis=1
)  # (6294, 3, 2, 44)


# %%%%%%%%%%%%%%
# Compute intersection of unit ray in world coordinates with mesh
# Use trimesh to compute intersection
locations, index_ray, index_tri = mesh.ray.intersects_location(
    ray_origins=[camera_extrinsics[frame_num][:3, 3]],
    ray_directions=[unit_ray_world[0, :, 0, 0]],
)

print(locations.shape)

# %%%%%%%%%%%%%%
# Compute intersection of unit ray in world coordinates with mesh
# for all frames
list_frame_nums = sorted(list(camera_extrinsics.keys()))

list_points_3d_world_individuals = []
for indiv in range(len(ds.individuals)):
    list_points_3d_world = []
    for frame_num in list_frame_nums:
        unit_ray_world_one = unit_ray_world[frame_num - 1, :, 0, indiv]
        if np.isnan(unit_ray_world_one).any():
            list_points_3d_world.append(np.nan * np.ones((1, 3)))
            continue

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=[camera_extrinsics[frame_num][:3, 3]],
            ray_directions=[unit_ray_world_one],
        )

        if locations.shape[0] == 0:
            print(f"No intersection for frame {frame_num}")
            list_points_3d_world.append(np.nan * np.ones((1, 3)))
            continue

        list_points_3d_world.append(locations)

    # concatenate all points for this individual
    points_3d_world_one = np.concatenate(list_points_3d_world)
    list_points_3d_world_individuals.append(points_3d_world_one)

points_3d_world = np.stack(list_points_3d_world_individuals, axis=-1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# next:
# - fit plane to mesh and compute transform to plane

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot mesh vertices (as points for simplicity)
# ax.scatter(
#     mesh.vertices[:, 0],
#     mesh.vertices[:, 1],
#     mesh.vertices[:, 2],
#     s=0.1,
#     color="gray",
# )

# # Plot ray
# origin = camera_extrinsics[frame_num][:3, 3]
# direction = points_ray_world[0, :, 0, 0]
# length = 10  # Adjust as needed
# ray_points = np.stack([origin, origin + direction * length])
# ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], color='red')

# Plot 3D points
ax.scatter(
    points_3d_world[:, 0, :].flatten(),
    points_3d_world[:, 1, :].flatten(),
    points_3d_world[:, 2, :].flatten(),
    color="blue",
)

# Plot camera poses as a trihedron
for frame_num in sorted(list(camera_extrinsics.keys()))[:1]:  # [0:-1:10]:
    R = camera_extrinsics[frame_num][:3, :3]
    center = camera_extrinsics[frame_num][:3, 3]
    axis_length = 1  # Adjust for visibility

    # X axis (red)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        R[0, 0],  # first column of rotation matrix
        R[1, 0],
        R[2, 0],
        color="r",
        length=axis_length,
        normalize=True,
    )
    # Y axis (green)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        R[0, 1],  # second column of rotation matrix
        R[1, 1],
        R[2, 1],
        color="g",
        length=axis_length,
        normalize=True,
    )
    # Z axis (blue)
    ax.quiver(
        center[0],
        center[1],
        center[2],
        R[0, 2],  # third column of rotation matrix
        R[1, 2],
        R[2, 2],
        color="b",
        length=axis_length,
        normalize=True,
    )

ax.set_aspect("equal")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
# %%
