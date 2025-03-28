# %%%%%%%%%%%%%%%%%%%

# Imports

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from movement.io import load_poses
from movement.kinematics import compute_speed
from movement.utils.vector import compute_norm, convert_to_unit

import matplotlib
%matplotlib widget

# %%%%%%%%%%%%%%%%%%%
# Input data

data = "/Users/sofia/swc/project_zebras/zebras-stitching/data/20250325_2228_id_unwrapped.h5"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read as movement dataset
ds = load_poses.from_sleap_file(data)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute body vectors

body_vector = ds.position.sel(keypoints="H") - ds.position.sel(keypoints="T")

# Select body vectors for which norm is outside mean +- 2 std
body_length = compute_norm(body_vector)

body_length_std = body_length.std()
body_length_mean = body_length.mean()
body_length_median = body_length.median()

body_vector_filtered = body_vector.where(
    np.logical_and(
        body_length <= body_length_mean + 2 * body_length_std,
        body_length >= body_length_mean - 2 * body_length_std,
    )
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot body length histogram per individual
fig, ax = plt.subplots()
counts, bins, _ = ax.hist(np.unstack(body_length.values, axis=-1), bins=100)
ax.vlines(
    body_length.mean(),
    ymin=0,
    ymax=np.max(counts),
    color="red",
    linestyle="-",
    label="mean body length",
)
ax.vlines(
    body_length.median(),
    ymin=0,
    ymax=np.max(counts),
    color="red",
    linestyle="-",
    label="mean body length",
)
ax.set_xlabel("body length (pixels)")
ax.set_ylabel("counts")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot body length histogram 2D
n_x_ticks = 10

fig, ax = plt.subplots()
im = ax.matshow(counts)


ax.set_aspect("equal")
ax.set_xticks(np.linspace(0, len(bins), n_x_ticks))
ax.set_xticklabels(["{:.2f}".format(b) for b in bins[0:-1:n_x_ticks]], rotation=90)
ax.set_xlabel("body length (pixels)")
ax.set_ylabel("track ID")

cbar = plt.colorbar(im)
cbar.set_label("counts")

print(counts.shape)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute average body vector per frame
body_vector_avg = body_vector_filtered.mean("individuals")
print(body_vector_avg.shape)

# Compute average **unit** body vector per frame
# (if unit, average is the same as resulting vector)
body_vector_unit_avg = convert_to_unit(body_vector_filtered).mean("individuals")
print(body_vector_unit_avg.shape)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute circular variance
# 1 - mean(resultant unit length)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html

circ_variance = 1 - compute_norm(body_vector_unit_avg)

fig, ax = plt.subplots()
ax.plot(circ_variance)
ax.set_xlabel("frame")
ax.set_ylabel("circular variance")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Polarisation of the herd
# mean(resultant unit length)

centroid_speed_avg = compute_speed(ds.position.mean("keypoints")).mean("individuals")

# plot polarisation and color by log of mean centroid-speed
fig, ax = plt.subplots()
sc = ax.scatter(
    x=compute_norm(body_vector_unit_avg).time,
    y=compute_norm(body_vector_unit_avg),
    c=np.log10(centroid_speed_avg),
    s=5
)
cbar = plt.colorbar(sc)
cbar.set_label("log10 avg centroid speed (pixels/frame)")

ax.set_xlabel("frame")
ax.set_ylabel("norm of mean body unit vector")
ax.set_title("polarisation")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot average body vector on the centroid trajectory

herd_centroid = ds.position.mean(["keypoints", "individuals"])

cmap = matplotlib.colormaps["viridis"]
values = np.linspace(0, 1, herd_centroid.shape[0])
colors = cmap(values)

fig, ax = plt.subplots()
# plot average body vector
qv = ax.quiver(
    herd_centroid.sel(space="x"),
    herd_centroid.sel(space="y"),
    body_vector_avg.sel(space="x"),
    body_vector_avg.sel(space="y"),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=4,
    headlength=5,
    headaxislength=5,
    # label="Head-to-snout vector",
    # scale=1,
    color=colors,
)
cbar = plt.colorbar(qv)
cbar.set_label("frame")
cbar.set_ticks(np.linspace(0, 1, 7))
cbar.set_ticklabels(np.arange(herd_centroid.shape[0]+1)[::int(herd_centroid.shape[0]/6)])

# plot one individual
indiv = "track_0"
ax.quiver(
    ds.position.mean("keypoints").sel(space="x", individuals=indiv),
    ds.position.mean("keypoints").sel(space="y", individuals=indiv),
    body_vector_filtered.sel(space="x",individuals=indiv),
    body_vector_filtered.sel(space="y",individuals=indiv),
    angles="xy",
    scale=1,
    scale_units="xy",
    headwidth=4,
    headlength=5,
    headaxislength=5,
    # label="Head-to-snout vector",
    # scale=1,
    color='k',
)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.set_title(f"average & {indiv} body vector")

# mark frame 4000
ax.scatter(
    herd_centroid.sel(space="x").isel(time=4000),
    herd_centroid.sel(space="y").isel(time=4000),
    marker="o",
    color="red",
    label="frame 4000",
)
ax.scatter(
    ds.position.mean("keypoints").sel(space="x", individuals=indiv, time=4000),
    ds.position.mean("keypoints").sel(space="y", individuals=indiv, time=4000),
    marker="o",
    color="red",
    label="frame 4000",
)

# plot alignment of one individual with the average body vector
track_0_cos = np.diag(
    np.dot(
        convert_to_unit(body_vector_filtered).sel(individuals=indiv), 
        body_vector_unit_avg.values.T
    )
)
fig, ax = plt.subplots()
ax.plot(
    track_0_cos
)
ax.vlines(
    4000,
    ymin=-1,
    ymax=1,
    color="red",
    linestyle="-",
    label="frame 4000",
)
ax.set_ylim(-1, 1)
ax.set_xlabel("frame")
ax.set_ylabel("dot with average unit body vector")
ax.set_title(f"alignment of {indiv} with average unit body vector")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%

# # body vectors x,y components for all individuals
# fig, ax = plt.subplots()
# body_vector_filtered.plot.line(x="time", row="space", aspect=2, size=2.5)

# # average body vector x,y components
# body_vector_avg.plot.line(x="time", row="space", aspect=2, size=2.5, color="red")

# # norm of average body vector
# fig, ax = plt.subplots()
# compute_norm(body_vector_avg).plot.line(x="time")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute alignment with average body vector

# OJO!! body_vector_avg_unit != body_vector_unit_avg
# body_vector_avg_unit = convert_to_unit(body_vector_avg)
body_vector_filtered_unit = convert_to_unit(body_vector_filtered)

# Compute dot product
cos_body_vector = xr.dot(
    body_vector_filtered_unit,
    body_vector_unit_avg, # != body_vector_avg_unit,
    dims=["space"],
)

# %%
np.testing.assert_almost_equal(track_0_cos, cos_body_vector.sel(individuals=indiv).values)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ax = plt.subplots()
im = ax.matshow(
    cos_body_vector,
    aspect="auto",
    cmap="coolwarm",
)
cbar = plt.colorbar(im)
cbar.set_label("alignment with average unit body vector")
ax.get_images()[0].set_clim(-1, 1)
ax.set_xlabel("individuals")
ax.set_ylabel("frame")

# %%
# # %%
# fig, ax = plt.subplots()
# body_vector_unit_avg.plot.line(x="time")
# %%
