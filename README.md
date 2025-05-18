# zebras-stitching
Exploring how to stitch zebra data

## Installation

From the root of the repository:

```bash
conda create -n zebras-env python=3.11 -y
conda activate zebras-env
pip install -r requirements.txt
```


## Notebooks

This repository explores possible pipelines for "unwrapping" trajectories of moving animals recorded with a camera drone, using open-source freely available software and off-the-shelf models. We use "unwrapping" to refer to the process of transforming the trajectories of the animals from a moving image coordinate system to a world coordinate system that is fixed to the ground. As a proof-of-concept, we focus on its application to a single video of 44 zebras in escape behavior.

The code is not meant as a fully-fledged package, but rather as a collection of notebooks to explore three different approaches and their trade-offs. We hope these notebooks can serve as a starting point for researchers to get familiar with the problem and the existing tools, or for further development of a more robust pipeline.

The input data consists on a set of 44 trajectories of zebras in escape behaviour, tracked in a moving image coordinate system (associated to the camera drone). These data was provided by the researchers as a SLEAP file (`20250325_2228_id.slp`). The three approaches we explore to compute the unwrapped trajectory are:

1. Using an image registration algorithm (via `itk-elastix`) to compute the transform from each frame to the previous frame.
2. Using a Structure-from-Motion (SfM) algorithm (via `opendronemap` and `opensfm`) to compute the camera poses at selected keyframes, and then interpolating the camera poses in the missing frames using slerp for rotations and linear interpolation for translations.
3. Using an SfM algorithm (via `opendronemap` and `opensfm`) to compute the camera poses at selected keyframes, and then interpolating the camera poses in the missing frames using `itk-elastix`'s image registration algorithm for rotations, and linear interpolation for translations.

To compare the performance of these three prototype approaches, we detect and track trees (using `deepforest` and `boxmot`) in the moving image coordinate system, and then unwrap the trajectories following each of the methods described above. Trees are useful since they are distinctive elements of the scene and they should be static relative to the ground. We evaluate the performance of each approach by computing the mean distance of each tree trajectory to its centroid in the world coordinate system.

Once the trajectories of the animals are "unwrapped" (i.e., in the world coordinate system), we demonstrate the applicability of these methods by computing common metrics from collective behaviour. We use  `movement` to clean the trajectories and programmatically correct ID swaps first, and then to compute these metrics of interest. 

We use Blender to generate custom orthophotos of the scene and overlay the generated 3D trajectories of the animals.

The notebooks are organized as follows:
- `00_notebook_elastix_transforms.py`: extracts the transforms from the image registration algorithm for approach 1.
- `01_notebook_elastix_unwrap_keypoints.py`: uses the transforms from notebook `00` to unwrap the input 2D trajectories (zebras or trees).
- `02_notebook_clean_unwrapped_tracks.py`: uses `movement` to clean the unwrapped trajectories of zebras.
- `03_notebook_compute_behaviour_metrics.py`: computes collective behaviour metrics of interest from the cleaned unwrapped zebra trajectories.
- `04_notebook_deepforest.py`: uses `deepforest` to detect trees in the image coordinate system, and the BotSort tracking algorithm (from `boxmot`) to track them.
- `05_notebook_reliable_trees.py`: uses  `movement` to select a subset of tree trajectories that have no ID swaps and low jitter, and which span the full area covered by the drone.
- `06_notebook_extract_sfm_keyframes.py`: it uses the output from the SfM algorithm (ran on the data using `opendronemap`) to extract the estimated camera intrinsic matrix, and the camera poses at the selected keyframes (one every 20 frames, plus the first and last frames). This notebook uses the `opensfm` library, which we install via the `opendronemap` container. However, the extracted transforms are exported to a csv file that can be used in subsequent notebooks.
- `07_notebook_interpolate_sfm.py`: it defines the camera poses at every frame by interpolating the camera poses at the keyframes using slerp for rotations and linear interpolation for translations.
- `08_notebook_apply_interpolated_sfm.py`: it applies the interpolated camera poses (either linearly or using `itk-elastix`'s image registration algorithm) to unwrap the trajectories of the zebras or trees.
- `09_notebook_export_3d_trajs_csv.py`: it exports the 3D trajectories of each invidual zebra as a csv file. These files are meant to be imported in Blender for visualisation along the OpenDroneMap generated mesh.
- `10_notebook_compute_tree_error.py`: it computes the distance of each tree trajectory to its centroid in the world coordinate system, and the weighted mean across all trees (weighted by the number of samples).
- `compute_orthophoto_blender.py`: is a script that is meant to be run using Blender's Python interpreter. It loads the OpenDroneMap generated mesh and the 3D trajectories of the animals into Blender, and generates a orthophoto of the scene with the virtual camera's z-axis parallel to the normal of the mesh best-fitting plane.

## OpenDroneMap pipeline
We used OpenDroneMap's command-line interface ([ODM](https://github.com/OpenDroneMap/ODM)) to run the SfM pipeline and generate the mesh. We used the GPU-compatible docker image and ran the pipeline on the 314 keyframes of the video (one every 20 frames, plus the first and last frames). In each of the keyframes, we provided masks around the individual zebras to prevent those pixels from being used in the reconstruction. 

We use the default setting except for the following custom flags: `--dem-resolution 1.0 --orthophoto-resolution 1.0 --pc-quality high`

## Toolkit
We would like to acknowledge the following tools and libraries that we used in this project:

- [itk-elastix](https://github.com/InsightSoftwareConsortium/ITKElastix)
- [elastix](https://elastix.dev/)
- [opendronemap](https://github.com/OpenDroneMap)
- [opensfm](https://github.com/mapillary/OpenSfM)
- [deepforest](https://github.com/weecology/DeepForest)
- [boxmot](https://github.com/mikel-brostrom/boxmot)
- [movement](https://github.com/neuroinformatics-unit/movement)
- [Blender](https://www.blender.org/)
- [SLEAP](https://sleap.ai/)
