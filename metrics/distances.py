"""Compute distances between zebras
===================================

Compute inter-individual distances between zebras moving in a group.
"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from movement.io import load_poses
from movement.plots import plot_centroid_trajectory
from movement.kinematics import compute_pairwise_distances
from movement.filtering import rolling_filter
from movement.transforms import scale
from movement.utils.vector import compute_norm

# %%
# Load unwrapped poses
# --------------------
# First let's define paths

repo_root = Path(__file__).parents[1]
data_dir = repo_root / "data"
video_dir = repo_root / "videos"
assert data_dir.exists()
assert video_dir.exists()

filename = "20250325_2228_id_unwrapped.h5"
file_path = data_dir / filename
video_path = video_dir / "21Jan_007.mp4"
background_path = video_dir / "21Jan_007_unwrapped_bacground.png"
for path in [file_path, video_path, background_path]:
    assert path.exists()

# %%
# Now, let's load the unwrapped poses
poses = load_poses.from_file(file_path, source_software="SLEAP", fps=30)

# %%
# Let' overlay them on the unwrapped background

fig, ax = plt.subplots(1, 1)
# Overlay an image of the experimental arena
ax.imshow(plt.imread(background_path))

# Plot centroid trajectories of the individuals
individuals = poses.individuals.values
# Cycle colors from the tab10 colormap
cmap = ["r", "g", "b", "c", "m", "y"]
colors = [
    cmap[i % len(cmap)] for i in range(len(individuals))
]

for id, color in zip(individuals, colors):
    plot_centroid_trajectory(
        poses.position,
        individual=id,
        ax=ax,
        linestyle="-",
        marker=".",
        s=0.1,
        c=color,
        alpha=0.2,
        label=id,
    )
ax.set_title("Individual trajectories within the arena")

# %%
# Scale position by body length
# -----------------------------

positions = poses.position
# Define the body length of the zebras in pixels
body_axis = positions.sel(keypoints="T") - positions.sel(keypoints="H")
body_length = compute_norm(body_axis)
# Compute the median body length (across time and individuals)
median_body_length = body_length.median(skipna=True).values
print(f"Median body length: {median_body_length:.2f} pixels")

# Scale the positions by the body length
positions_scaled = scale(
    positions,
    factor=1 / median_body_length,
    space_unit="body lengths",
)



# %%
# Compute pairwise distances
# --------------------------

# Compute the centroid of each zebra
centroids = positions_scaled.mean(dim="keypoints", skipna=True)

# Generate a dict mapping from pair names to data arrays containing
# inter-centroid distances over time for that pair of individuals
distances_dict = compute_pairwise_distances(
    centroids,
    dim="individuals",
    pairs="all",
    metric="euclidean",
)

# %%
# Let's stack the dict of distances into a single data array
distances = xr.concat(
    distances_dict.values(),
    dim="id_pair",
)
distances = distances.assign_coords(
    id_pair=list(distances_dict.keys())
)
distances.name = "Distance (body lengths)"
print(distances)


# %%
# Plot all pair-wise distances across time as a heatmap

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
distances.plot(ax=ax)
# Don't show the y labels
ax.set_yticklabels([])
ax.set_title("Pair-wise distances between zebras")
ax.set_xlabel("Time (s)")


# %%
# Plot min, max, and mean distances between pairs of zebras across time

fig, ax = plt.subplots(1, 1)
distances.mean(dim="id_pair").plot(label="mean", ax=ax)
distances.min(dim="id_pair").plot(label="min", ax=ax)
distances.max(dim="id_pair").plot(label="max", ax=ax)
ax.legend()
ax.set_title("Inter-individual distances")


# %%
# Nearest neighbors
# -----------------

# Creat an array of NaN with shape (time, individuals)
distances_nn = xr.DataArray(
    data=np.nan,
    dims=["time", "individuals"],
    coords=dict(
        time=poses.time,
        individuals=poses.individuals,
    ),
)

distances_nn.name = "Distance to Nearest Neighbor (pixels)"

# Compute the nearest neighbor distance for each individual
for id in individuals:
    pairs_with_id = [pair for pair in distances_dict.keys() if id in pair]
    distances_nn.loc[dict(individuals=id)] = distances.sel(
        id_pair=pairs_with_id).min(
            dim="id_pair", skipna=True
        )

print(distances_nn)

# %%
# Plot the nearest neighbor distances across time
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
distances_nn.transpose().plot(ax=ax)
# Don't show the y labels
ax.set_yticklabels([])
ax.set_title("Distance to nearest neighbor")
ax.set_xlabel("Time (s)")


# %%
# Plot mean, mix, and max nearest neighbor distances across time
fig, ax = plt.subplots(1, 1)
distances_nn.mean(dim="individuals").plot(label="mean", ax=ax)
distances_nn.min(dim="individuals").plot(label="min", ax=ax)
distances_nn.max(dim="individuals").plot(label="max", ax=ax)
ax.legend()
ax.set_title("Nearest neighbot distances (body lengths)")
ax.set_xlabel("Time (s)")

# %%
