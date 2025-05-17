# %%
# imports
from pathlib import Path


# %%
# Input data
repo_root = Path(__file__).parent
data_dir = repo_root / "data"
assert data_dir.exists()

# tree data per method
paths_tree_data = {
    "itk-all": (
        data_dir
        / "approach-itk-all"
        / "20250325_2228_id_unwrapped_20250403_161408_clean.h5"
    ),
    "sfm-pcs-2d": (
        data_dir
        / "approach-sfm-interp"
        / "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5"
    ),
}

# clean zebra data per method
paths_zebra_data = {
    "itk-all": (
        data_dir
        / "approach-itk-all"
        / "20250325_2228_id_unwrapped_20250403_161408_clean.h5"
    ),
    "sfm-pcs-2d": (
        data_dir
        / "approach-sfm-interp"
        / "20250325_2228_id_sfm_interp_PCS_2d_20250516_155745_clean.h5"
    ),
}


# %%
# Compute median body length per method


# %%
# Compute standard deviation of tree positions per method
# normalized by median body length


# %%
# Export results as latex table
