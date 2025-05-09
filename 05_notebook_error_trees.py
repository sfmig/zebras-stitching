# %%
# Imports

from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
from movement.io import load_bboxes

# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

filename = "data/21Jan_007_tracked_trees_20250505_100631.csv"

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read VIA-tracks file as a movement dataset

ds = load_bboxes.from_via_tracks_file(
    file_path=filename, use_frame_numbers_from_file=False
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get length of trajectory (ie number of samples that are not nan)

length_trajectory = np.array(
    [
        len(
            list(
                next(groupby(np.all(~np.isnan(ds.position.values), axis=1)[:, ind]))[1]
            )
        )
        for ind in range(len(ds.individuals))
    ]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Total variation in x and y positions

# Linearly interpolate position first?

total_variation_x = np.nansum(
    np.abs(np.diff(ds.position.sel(space="x"), axis=0)), axis=0
)
total_variation_y = np.nansum(
    np.abs(np.diff(ds.position.sel(space="y"), axis=0)), axis=0
)

# add and normalize by length of trajectory
total_variation = (total_variation_x + total_variation_y) / length_trajectory

# Remove values with 0 total variation
total_variation[total_variation == 0] = np.nan


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot total variation in x and y per individual

fig, ax = plt.subplots()

ax.plot(total_variation_x, label="total variation in x-coordinate")
ax.plot(total_variation_y, label="total variation in y-coordinate")

ax.set_xlabel("tree")
ax.set_ylabel("total variation (pixels)")
ax.legend()


fig, ax = plt.subplots()

ax.plot(total_variation, label="total variation normalised by length of trajectory")   

ax.set_xlabel("tree")
ax.set_ylabel("total variation (pixels)")
ax.legend()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Sort by total variation
# (We would like trajectories with LOW total variation)
total_variation_no_nan = total_variation[~np.isnan(total_variation)]
ind_sorted = np.argsort(total_variation_no_nan)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot x,y positions

fig, ax = plt.subplots()


# plot first 10 trajectories (with MOST total variation)
ax.plot(
    ds.position.sel(space="x").isel(individuals=ind_sorted[:100]),
    ds.position.sel(space="y").isel(individuals=ind_sorted[:100]),
    label=ind_sorted[:100],
)  
ax.legend()

fig, ax = plt.subplots()
# plot last 10 trajectories (with MOST total variation)
ax.plot(
    ds.position.sel(space="x").isel(individuals=ind_sorted[-100:]),
    ds.position.sel(space="y").isel(individuals=ind_sorted[-100:]),
)  # ds.position.sel(space="y"), ".")


# %%
