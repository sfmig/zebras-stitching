"""
Detect and track trees in a video using DeepForest

"""
# %%
from deepforest import main
from deepforest import get_data
from deepforest.visualize import plot_annotations, plot_results, plot_predictions

from matplotlib import pyplot as plt
import sleap_io as sio
from pathlib import Path
# %%
# %matplotlib widget
# %%
# Data
video_path = Path("/home/sminano/swc/project_zebras/videos/21Jan_007.mp4")

# %%
video_array = sio.load_video(video_path)


# %%
# Initialize the model class
model = main.deepforest()

# Load a pretrained tree detection model from Hugging Face
model.load_model(model_name="weecology/deepforest-tree", revision="main")

# %%
list_df = []
for im in video_array[:5]:
    df = model.predict_image(image=im)
    list_df.append(df)
# %%
# plot
# fig, ax = plt.subplots(1, 1)
# ax.imshow(im)
# img = plot_predictions(im, df)
plot_results(df, image=im)
# ax.imshow(img)
# %%
