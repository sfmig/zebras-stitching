"""Clean noisy zebra tracks
===========================

Load messy SLEAP predictions of zebra tracks and try to clean them.
"""

# %%
# Imports
# -------
from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

from movement.filtering import filter_by_confidence
from movement.io import load_poses, save_poses
from movement.kinematics import compute_displacement
from movement.utils.reports import report_nan_values
from movement.utils.vector import compute_norm

# %%
# Load the data
# -------------

data_dir = Path.home() / "Data" / "zebras_sleap"
filename = data_dir / "270Predictions.slp"
ds = load_poses.from_sleap_file(filename, fps=30)

print(ds)


# %%

ds.position.squeeze().plot(
    x="time", row="keypoints", hue="space", aspect=2
)

# %%
# Filter by confidence
# --------------------

# %%
# Plot a histogram of confidence values
colors = ["green", "purple"]

for i, keypoint in enumerate(ds.keypoints.values):
    ds.confidence.sel(keypoints=keypoint).plot.hist(
        bins=50, histtype="step", color=colors[i], label=keypoint
    )
    plt.legend(title="Keypoint")
    plt.title("Confidence histogram")

# %%
# Filter out low-confidence keypoints

position_filt = filter_by_confidence(
    ds.position, ds.confidence, threshold=0.9
)


# %%
# Plot the filtered keypoints
position_filt.squeeze().plot(
    x="time", row="keypoints", hue="space", aspect=2
)

# We actually got rid of most of the erroneous keypoints!
# But it's not enough, we'll add 1 more filter

# %%
# Filter by distance
# -------------------

# We will filter out keypoints that are too far from the previous position
# (zebras can't teleport!)

# Plot a histogram of distances frame-to-frame
displacement = compute_displacement(position_filt)
distance = compute_norm(displacement)
for i, keypoint in enumerate(distance.keypoints.values):
    distance.sel(keypoints=keypoint).squeeze().plot.hist(
        bins=50, histtype="step", color=colors[i], label=keypoint
    )
    plt.legend()
    plt.title("Distance histogram")

# %%
# Let's right a function to do this


def filter_by_distance(
    position: xr.DataArray,
    threshold: float = 10.0,
    print_report: bool = True,
) -> xr.DataArray:
    """Filter out keypoints that are too far from the previous position.

    This function first computes the displacement array, which is defined as
    the difference between the position array at time point ``t`` and the
    position array at time point ``t-1``.

    As a result, for a given individual and keypoint, the displacement vector
    at time point ``t``, is the vector pointing from the previous
    ``(t-1)`` to the current ``(t)`` position, in cartesian coordinates.
    For the 1st time point, the displacement is set to 0.

    The magnitude of the displacement vector is the Euclidean distance
    between the 2 points. We drop values where that distance is greater than
    the given ``threshold``.

    Parameters
    ----------
    position : xr.DataArray
        The position array to filter. It should have the dimensions
        ``("individuals", "time", "keypoints", "space")``.
    threshold : float, optional
        The maximum distance allowed between 2 consecutive positions.
        Defaults to 100.0.
    print_report : bool, optional
        Whether to print a report of the number of NaN values before and after
        filtering. Defaults to True.

    Returns
    -------
    xr.DataArray
        The filtered position array.

    See Also
    --------
    movement.kinematics.compute_displacement
    movement.utils.vector.compute_norm
    movement.utils.reports.report_nan_values

    """
    displacement = compute_displacement(position)
    distance = compute_norm(displacement)
    # Set all values to NaN where the distance is greater than the threshold
    position_filtered = position.where(distance < threshold)

    if print_report:
        print(report_nan_values(position, "input"))
        print(report_nan_values(position_filtered, "output"))
    return position_filtered


# %%
# Apply the filter

position_clean = filter_by_distance(position_filt, threshold=10)

# %%
# Plot the filtered keypoints
position_clean.squeeze().plot(
    x="time", row="keypoints", hue="space", aspect=2
)

# %%
# Plot the raw data again for comparison
ds.position.squeeze().plot(
    x="time", row="keypoints", hue="space", aspect=2
)
# %%

# Save the cleaned data
# ---------------------

ds_clean = xr.Dataset(
    {
        "position": position_clean,
        "confidence": ds.confidence,
    }
)
ds_clean.attrs = ds.attrs.copy()

save_poses.to_dlc_file(
    ds_clean, data_dir / "270Predictions_clean.csv",
    split_individuals=False,
)

save_poses.to_sleap_analysis_file(
    ds_clean, data_dir / "270Predictions_clean.analysis.h5",
)

# %%
