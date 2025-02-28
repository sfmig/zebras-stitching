# %%
from pathlib import Path

import cv2 as cv
import numpy as np
import sleap_io as sio
from matplotlib import pyplot as plt

from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_estimator import CameraEstimator
from stitching.camera_wave_corrector import WaveCorrector
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.images import Images
from stitching.warper import Warper


# %%
# Utils
def plot_image(img, figsize_in_inches=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show()
    return fig, ax


def plot_images(imgs, figsize_in_inches=(5, 5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.show()
    return fig, axs


# %%
# Data paths

sleap_file = "/Users/sofia/swc/project_zebras/270Predictions.slp"
video_file = "/Users/sofia/swc/project_zebras/21Jan_007.mp4"


# %%
# Inspect video data

video = sio.load_video(video_file)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")

# %%
# Select a few frames for stitching and plot

frame_idcs = [0, 200, 300]

for i, frame_idx in enumerate(frame_idcs):
    frame = video[frame_idx]
    fig, ax = plot_image(frame, figsize_in_inches=(5, 5))
    ax.set_title(f"Frame {frame_idx}")


# %%
# Resize images
list_images = np.split(
    video[frame_idcs], len(frame_idcs), axis=0
)  # each image is a numpy array
list_images = [x.squeeze() for x in list_images]
original_imgs = Images.of(list_images)
print(original_imgs.sizes)

medium_res_imgs = list(original_imgs.resize(Images.Resolution.MEDIUM))
low_res_imgs = list(original_imgs.resize(Images.Resolution.LOW))
final_res_imgs = list(original_imgs.resize(Images.Resolution.FINAL))

plot_images(low_res_imgs, (20, 20))
# %%
# Print sizes
original_size = original_imgs.sizes[0]
medium_size = original_imgs.get_image_size(medium_res_imgs[0])
low_size = original_imgs.get_image_size(low_res_imgs[0])
final_size = original_imgs.get_image_size(final_res_imgs[0])


print(
    f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP"
)
print(
    f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP"
)
print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")

# %%
# Find features on the medium res image

finder = FeatureDetector("orb")  # , nfeatures=1000)
features = [finder.detect_features(img) for img in medium_res_imgs]

# visualise detected features
for i, img in enumerate(medium_res_imgs):
    features_on_img = finder.draw_keypoints(img, features[i])
    fig, ax = plot_image(features_on_img, (15, 10))
    ax.set_title(f"Detected features on frame {frame_idcs[i]}")

# %%
# Match features
matcher = FeatureMatcher(matcher_type="affine")
matches = matcher.match_features(features)


# %%
# Visualise matches
matcher.get_confidence_matrix(matches)


# %%
all_relevant_matches = matcher.draw_matches_matrix(
    medium_res_imgs,
    features,
    matches,
    conf_thresh=1,
    inliers=True,
    matchColor=(0, 255, 0),
)

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1 + 1} to Image {idx2 + 1}")
    plot_image(img, (20, 10))
# %%
# # Subset features with a good match
# from stitching.subsetter import Subsetter

# subsetter = Subsetter()
# indices = subsetter.get_indices_to_keep(features, matches)
# features = subsetter.subset_list(features, indices)
# %%
# Camera estimation
camera_estimator = CameraEstimator("affine")
camera_adjuster = CameraAdjuster("affine")
wave_corrector = WaveCorrector("auto")

cameras = camera_estimator.estimate(features, matches)

print([cam.focal for cam in cameras])
print([cam.R for cam in cameras])
print([cam.t for cam in cameras])
print([cam.K for cam in cameras])
# %%
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)
# %%
# Stitch low res images
warper = Warper()  # "plane"?
warper.set_scale(cameras)

low_sizes = original_imgs.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = original_imgs.get_ratio(
    Images.Resolution.MEDIUM, Images.Resolution.LOW
)  # since the cameras estimates were obtained on medium res imgs


warped_low_imgs = list(warper.warp_images(low_res_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
print(low_corners)
print(low_sizes)

# %%
plot_images(warped_low_imgs, (10, 10))
plot_images(warped_low_masks, (10, 10))

# %%
# Stitch final res images
final_sizes = original_imgs.get_scaled_img_sizes(Images.Resolution.FINAL)
camera_aspect = original_imgs.get_ratio(
    Images.Resolution.MEDIUM, Images.Resolution.FINAL
)

warped_final_imgs = list(
    warper.warp_images(
        final_res_imgs,
        cameras,
        camera_aspect,
    )
)
warped_final_masks = list(
    warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
)
final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)
# %%
print(final_corners)
print(final_sizes)

