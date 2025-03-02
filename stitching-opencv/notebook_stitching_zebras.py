"""Stitching of zebra images using OpenCV.

Follows https://github.com/OpenStitching/stitching_tutorial/blob/master/docs/Stitching%20Tutorial.md

TODO:
- get translation vectors
- mask bboxes around zebras
"""

# %%%%%%%%%%%%%%

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
from stitching.timelapser import Timelapser
from stitching.warper import Warper

# %matplotlib widget


# %%%%%%%%%%%%%%%%%%%
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


# %%%%%%%%%%%%%%%%%%%
# Data paths

sleap_file = "/Users/sofia/swc/project_zebras/270Predictions.slp"
video_file = "/Users/sofia/swc/project_zebras/21Jan_007.mp4"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Inspect video data

video = sio.load_video(video_file)
n_frames, height, width, channels = video.shape

print(f"Number of frames: {n_frames}")
print(f"Frame size: {width}x{height}")
print(f"Number of channels: {channels}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select a few frames for stitching and plot

frame_idcs = [0, 50, 100]

for i, frame_idx in enumerate(frame_idcs):
    frame = video[frame_idx]
    fig, ax = plot_image(frame, figsize_in_inches=(5, 5))
    ax.set_title(f"Frame {frame_idx}")

list_images = np.split(
    video[frame_idcs], len(frame_idcs), axis=0
)  # each image is a numpy array
list_images = [x.squeeze() for x in list_images]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define masks
# define masks: 255 is enabled, 0 is disabled
list_masks = [np.ones_like(img)[:, :, 0] * 255 for img in list_images]

# top:bottom, left:right -- areas to mask!
list_masks[0][300:630, 900:1217] = 0
list_masks[1][300:630, 850:1165] = 0
list_masks[2][250:600, 850:1170] = 0


# plot masks (2d arrays) on images
masked_imgs = [cv.bitwise_and(img, img, mask=mask) for img, mask in zip(list_images, list_masks)]
plot_images(masked_imgs, (20, 20))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Resize images

original_imgs = Images.of(list_images)
original_masks = Images.of(list_masks)
print(original_imgs.sizes)

medium_res_imgs = list(original_imgs.resize(Images.Resolution.MEDIUM))
low_res_imgs = list(original_imgs.resize(Images.Resolution.LOW))
final_res_imgs = list(original_imgs.resize(Images.Resolution.FINAL))

medium_res_masks = list(original_masks.resize(Images.Resolution.MEDIUM))
low_res_masks = list(original_masks.resize(Images.Resolution.LOW))
final_res_masks = list(original_masks.resize(Images.Resolution.FINAL))

plot_images(low_res_imgs, (20, 20))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print sizes
original_size = original_imgs.sizes[0]
medium_size = original_imgs.get_image_size(medium_res_imgs[0])
low_size = original_imgs.get_image_size(low_res_imgs[0])
final_size = original_imgs.get_image_size(final_res_imgs[0])


print(f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px")
print(f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px")
print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px")
print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Find features on the medium res image -- it should avoid zebras

finder = FeatureDetector("orb")  # , nfeatures=1000)
features = finder.detect_with_masks(medium_res_imgs, medium_res_masks)

# visualise detected features
for i, img in enumerate(medium_res_imgs):
    features_on_img = finder.draw_keypoints(img, features[i])
    fig, ax = plot_image(features_on_img, (15, 10))
    ax.set_title(f"Detected features on frame {frame_idcs[i]}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Match features
matcher = FeatureMatcher(matcher_type="affine")
matches = matcher.match_features(features)

# Visualise matches
print(matcher.get_confidence_matrix(matches))


# Draw links between matches
all_relevant_matches = matcher.draw_matches_matrix(
    medium_res_imgs,
    features,
    matches,
    conf_thresh=1.61,
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
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Camera estimation
camera_estimator = CameraEstimator("affine")
camera_adjuster = CameraAdjuster("affine")
# wave_corrector = WaveCorrector("auto") -  not required for affine

cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
# cameras = wave_corrector.correct(cameras)

print([cam.focal for cam in cameras])
print([cam.R for cam in cameras])
print([cam.t for cam in cameras])
print([cam.K for cam in cameras])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Stitch low res images
warper = Warper("affine")
warper.set_scale(cameras)

low_sizes = original_imgs.get_scaled_img_sizes(Images.Resolution.LOW)
camera_aspect = original_imgs.get_ratio(
    Images.Resolution.MEDIUM, Images.Resolution.LOW
)  # since the cameras estimates were obtained on medium res imgs


warped_low_imgs = list(warper.warp_images(low_res_imgs, cameras, camera_aspect))
warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))

plot_images(warped_low_imgs, (10, 10))
plot_images(warped_low_masks, (10, 10))


# With the warped corners and sizes
# we know where the images will be placed on the final plane
low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
print(low_corners)
print(low_sizes)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    warper.create_and_warp_masks(
        final_sizes,
        cameras,
        camera_aspect,
    )
)
final_corners, final_sizes = warper.warp_rois(
    final_sizes,
    cameras,
    camera_aspect,
)

plot_images(warped_final_imgs, (10, 10))
plot_images(warped_final_masks, (10, 10))

# With the FINAL warped corners and sizes
# we know where the images will be placed on the final plane
print(final_corners)
print(final_sizes)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Timelapser
timelapser = Timelapser("as_is")
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(warped_final_imgs, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (10, 10))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---------------------------
# Crop--->  Numba needs NumPy 2.1 or less. Got NumPy 2.2.

from stitching.cropper import Cropper

cropper = Cropper()

cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

cropped_low_masks = list(cropper.crop_images(warped_low_masks))
cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

lir_aspect = original_imgs.get_ratio(
    Images.Resolution.LOW, Images.Resolution.FINAL
)  # since lir was obtained on low imgs
cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Seamfinder

# from stitching.seam_finder import SeamFinder

# seam_finder = SeamFinder()

# seam_masks = seam_finder.find(warped_final_imgs, final_corners, warped_final_masks)
# seam_masks = [
#     seam_finder.resize(seam_mask, mask)
#     for seam_mask, mask in zip(seam_masks, warped_final_masks)
# ]

# seam_masks_plots = [
#     SeamFinder.draw_seam_mask(img, seam_mask)
#     for img, seam_mask in zip(warped_final_masks, seam_masks)
# ]
# plot_images(seam_masks_plots, (15, 10))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# All in one go
from stitching import AffineStitcher

print(AffineStitcher.AFFINE_DEFAULTS)  # overwrites some of Stitcher.DEFAULT_SETTINGS

settings = {
    "crop": False,  # The whole plan should be considered
    "confidence_threshold": 0.5,  # The matches confidences aren't that good
}

stitcher = AffineStitcher(**settings)
panorama = stitcher.stitch(list_images, list_masks)

plot_image(panorama, (20, 20))

# %%
# creating the image mask and setting only the middle 20% as enabled (255)
# height, width = img1.shape[:2]
# top, bottom, left, right = map(
#     int, (0.4 * height, 0.6 * height, 0.4 * width, 0.6 * width)
# )
# mask = np.zeros(shape=(height, width), dtype=np.uint8)
# mask[top:bottom, left:right] = 255
# mask = np.zeros(shape=(height, width), dtype=np.uint8)
# mask[top:bottom, left:right] = 255
