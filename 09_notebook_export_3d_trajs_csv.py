# %%
from movement.io import load_poses
from pathlib import Path
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

data_dir = Path("data")

# "uncleaned" trajectories
zebras_3d_file = data_dir / "20250325_2228_id_sfm_interp_WCS_3d_20250516_155745.h5"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create output directory
blender_csv_dir = data_dir / f"blender-csv-{zebras_3d_file.stem}"
blender_csv_dir.mkdir(exist_ok=False)  # do not overwrite existing files

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read 3D data as a movement dataset
ds = load_poses.from_sleap_file(zebras_3d_file)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export each individual's 3D trajectory as a CSV file

centroid_da = ds.position.mean(dim="keypoints")
for individual in ds.individuals.data:

    filepath = blender_csv_dir / f"{individual}_3d_traj.csv"

    centroid_individual = centroid_da.sel(individuals=individual).to_pandas() # for 2D data

    # remove rows with nans before exporting
    centroid_individual = centroid_individual.dropna()

    # write to csv
    centroid_individual.to_csv(filepath, index=False)


# %%
