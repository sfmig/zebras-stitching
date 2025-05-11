
"""Compute 3d trajectories using SFM output and mesh.

Requires launching this in the opendm container.
"""
# %%%%%%
from datetime import datetime
import numpy as np
import trimesh
import xarray as xr
from opensfm import dataset # requires launching this in the opendm container
from movement.io import load_poses, save_poses
from movement.io.validators import ValidPoseTracks
import matplotlib.pyplot as plt
from pathlib import Path
%matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
opensfm_dir = Path("/workspace/datasets/project/opensfm")
mesh_path = Path(
    "/workspace/datasets/project/odm_meshing/odm_25dmesh.ply"
)

points_2d_slp = Path(
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

# Replace the plane mesh creation and ray intersection with this mathematical solution:
def ray_plane_intersection(ray_origins, ray_directions_unit, plane_normal, plane_point):
    """
    Compute the intersection points of an array of rays with a plane.
    
    Parameters:
    ray_origins: (N, 3) array of ray origins
    ray_directions: (N, 3) array of ray directions (should be unit vectors)
    plane_normal: (3,) normal vector to the plane (should be unit vector)
    plane_point: (3,) point contained in the plane
    
    Returns:
    intersections: (N, 3) array of intersection points (NaN where no intersection)

    The mathematical solution uses the parametric form of the ray equation and 
    the point-normal form of the plane equation to find the intersection point. 
    The intersection point p fullfils the following two equations:
    - The ray equation:
        p = ray_origin + t * ray_direction
    - The plane equation:
        (p - plane_point) · plane_normal = 0
    We can combine these two equations to solve for the parameter t, 
    which we then use to find the intersection point p.
    
    The function also includes checks for:
    - Rays parallel to the plane (denominator close to zero)
    - Intersections behind the camera (t ≤ 0)
    - Invalid rays (NaN in input)
    """
    # Initialize array for intersections
    intersections = np.full_like(ray_origins, np.nan)
    
    # Compute projection of the ray vector onto the plane normal
    ray_projection = np.dot(ray_directions_unit, plane_normal)

    # Find rays that intersect the plane (projection != 0)
    valid_rays = np.abs(ray_projection) > 1e-10
    
    if np.any(valid_rays):
        # Compute t parameter for valid rays
        t = np.dot(plane_point - ray_origins[valid_rays], plane_normal) / ray_projection[valid_rays]
        
        # Only keep intersections in front of the camera (t > 0)
        valid_t = t > 0
        valid_rays[valid_rays] = valid_t
        
        # Compute intersection points
        intersections[valid_rays] = ray_origins[valid_rays] + t[valid_t, np.newaxis] * ray_directions_unit[valid_rays]
    
    return intersections
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read mesh and fit a plane to it
mesh = trimesh.load(mesh_path)

# Fit a plane to the mesh vertices
vertices = mesh.vertices
center = vertices.mean(axis=0)  # a point on the plane


# Get the covariance matrix
cov = np.cov(vertices.T)
# Get the eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eigh(cov)
# The normal vector is the eigenvector corresponding to the smallest eigenvalue
normal = eigenvecs[:, 0]



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
list_frame_idx = []
list_3D_points_per_frame = []
pt2D_pixels_homogeneous_shape = position_homogeneous.sel(time=0).shape 

for frame_filename in list_frame_filenames: 


    # Get intrinsic and extrinsic camera parameters for this frame
    shot = recs.shots[frame_filename]
    cam = shot.camera # instrinsic matrix?
    pose = shot.pose # extrinsic

    # Get 2D pixel & **homogeneous** coordinates of points at this frame
    # (M,3 array, with M = total number of points)
    # (pixel coordinates = not normalised)
    frame_idx = int(frame_filename.split(".")[0])
    list_frame_idx.append(frame_idx)
    pt2D_pixels_homogeneous = position_homogeneous.sel(time=frame_idx).values.reshape(-1,3)  

    # Compute *normalised* coordinates in camera coordinate system (aka bearings)
    pt2D_bearings_homogeneous_ccs = (
        np.linalg.inv(cam.get_K_in_pixel_coordinates(image_width, image_height)) 
        @ pt2D_pixels_homogeneous.T
    ).T  # returns normalised coordinates!

    # Compute depth from intersection of ray with mesh
    # 1- compute free vector in world coordinates
    bearings_rotated_to_wcs = (pose.get_R_cam_to_world() @ pt2D_bearings_homogeneous_ccs.T).T
    bearings_rotated_to_wcs = bearings_rotated_to_wcs / np.linalg.norm(bearings_rotated_to_wcs, axis=1, keepdims=True)

    # 2- compute intersection with mesh
    # exclude nans
    pt3D_world_w_nans = np.nan * np.ones_like(pt2D_pixels_homogeneous)
    slc_nan_bearings = np.isnan(bearings_rotated_to_wcs).any(axis=1)

    # In your main loop, replace the mesh.ray.intersects_location call with:
    pt3D_world_normalized = ray_plane_intersection(
        ray_origins=np.tile(pose.get_t_cam_to_world(), (sum(~slc_nan_bearings), 1)),
        ray_directions_unit=bearings_rotated_to_wcs[~slc_nan_bearings, :],
        plane_normal=normal,
        plane_point=center
    )

    # Reshape to original shape
    pt3D_world_w_nans[~slc_nan_bearings, :] = pt3D_world_normalized
    pt3D_world_w_nans = pt3D_world_w_nans.reshape(pt2D_pixels_homogeneous_shape)

    # append
    list_3D_points_per_frame.append(pt3D_world_w_nans)


# Concatenate all 3D points
pt3D_world_all = np.stack(list_3D_points_per_frame, axis=0)

print(pt3D_world_all.shape)  # frame, individual, kpt, space



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as movement 0.0.5 dataset

ds_export = xr.Dataset(
    data_vars=dict(
        pose_tracks=(ds.pose_tracks.dims, pt3D_world_all),
        confidence=(ds.confidence.dims, ds.confidence.sel(time=list_frame_idx).values),
    ),
    coords=dict(
        time=list_frame_idx,
        individuals=ds.individuals.values,
        keypoints=ds.keypoints.values,
        space=['x', 'y', 'z'],
    ),
    attrs={"source_file": ""},
)

# ds_export = ValidPoseTracks(ds_export)

# get string timestamp of  today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

slp_file = save_poses.to_dlc_file(
    ds_export,
    f"{Path(points_2d_slp).stem}_sfm_3d_unwrapped_{timestamp}.h5",
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as numpy array
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
np.savez(
    Path("data") / f"{Path(points_2d_slp).stem}_sfm_3d_unwrapped_{timestamp}.npz",
    position=pt3D_world_all,
    confidence=ds.confidence.sel(time=list_frame_idx).values,
    dimensions=np.array(['time', 'individuals', 'keypoints', 'space']),  # following position.shape
    time=list_frame_idx,
    individuals=ds.individuals.values,
    keypoints=ds.keypoints.values,
    space=['x', 'y', 'z'],
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color_per_frame = plt.cm.viridis(np.linspace(0, 1, pt3D_world_all.shape[0]))
color_per_individual = plt.cm.viridis(np.linspace(0, 1, pt3D_world_all.shape[1]))
for individual_idx in range(pt3D_world_all.shape[1]):
    for frame_idx in range(pt3D_world_all.shape[0]):
        # line
        ax.plot(
            pt3D_world_all[frame_idx, individual_idx, :, 0], 
            pt3D_world_all[frame_idx, individual_idx, :, 1], 
            pt3D_world_all[frame_idx, individual_idx, :, 2], 
            '-',
            label=f"frame {frame_idx}",
            c=color_per_frame[frame_idx]
        )
        # # head
        # ax.scatter(
        #     pt3D_world_all[frame_idx, individual_idx, 0, 0], 
        #     pt3D_world_all[frame_idx, individual_idx, 0, 1], 
        #     pt3D_world_all[frame_idx, individual_idx, 0, 2], 
        #     'o',
        #     c=color_per_individual[individual_idx]
        # )
        # # tail
        # ax.scatter(
        #     pt3D_world_all[frame_idx, individual_idx, 1, 0], 
        #     pt3D_world_all[frame_idx, individual_idx, 1, 1], 
        #     pt3D_world_all[frame_idx, individual_idx, 1, 2], 
        #     '*',
        #     c=color_per_individual[individual_idx]
        # )
    
# plt.legend()
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


