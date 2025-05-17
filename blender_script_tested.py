"""
Script to render a custom orthophoto using Blender.

Loads the mesh .obj file and represents the trajectories of each individual
as separate polylines. Creates a virtual camera and orients its z-axis parallel
to the normal of the mesh's best-fitting plane. It then renders the scene from
the camera position.

Blender version: 4.0.2

"""

from datetime import datetime
import bpy
import csv
from pathlib import Path

import numpy as np
import mathutils

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
data_dir = Path("/Users/sofia/swc/project_zebras/zebras-stitching/data")
csv_dir = data_dir / "blender-csv-20250325_2228_id_sfm_interp_WCS_3d_20250516_155745"

# mesh file
mesh_file = "/Users/sofia/swc/project_zebras/datasets_step20_03_0indexing_masking/project_no_images/odm_texturing/odm_textured_model_geo.obj"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Other input parameters

# offset added to the trajectories for visibility
# (the trajectories are defined in the best-fitting plane)
z_offset = 0.05


plane_normal = np.array([-0.08907009, 0.08261507, -0.9925932])
plane_center = np.array([1.24176788, 0.15539519, -1.00927566])

# the camera is placed at the plane centre plus this offset
camera_position_offset = np.array([0.40, -0.40, 4.75])

# Define colors array
colors_array = np.array(
    [
        [0.18995, 0.07176, 0.23217, 1.0],
        [0.21291, 0.12947, 0.37314, 1.0],
        [0.23582, 0.1972, 0.52373, 1.0],
        [0.25369, 0.26327, 0.65406, 1.0],
        [0.26652, 0.32768, 0.76412, 1.0],
        [0.27429, 0.39043, 0.85393, 1.0],
        [0.27701, 0.45152, 0.92347, 1.0],
        [0.27469, 0.51094, 0.97275, 1.0],
        [0.26252, 0.56967, 0.99773, 1.0],
        [0.23288, 0.62923, 0.99202, 1.0],
        [0.19326, 0.68812, 0.9619, 1.0],
        [0.15173, 0.74472, 0.91416, 1.0],
        [0.11639, 0.7974, 0.85559, 1.0],
        [0.09532, 0.84455, 0.79299, 1.0],
        [0.09662, 0.88454, 0.73316, 1.0],
        [0.12733, 0.91701, 0.67627, 1.0],
        [0.18491, 0.94484, 0.60713, 1.0],
        [0.2618, 0.96765, 0.52981, 1.0],
        [0.35043, 0.98477, 0.45002, 1.0],
        [0.44321, 0.99551, 0.37345, 1.0],
        [0.53255, 0.99919, 0.30581, 1.0],
        [0.61088, 0.99514, 0.2528, 1.0],
        [0.66428, 0.98524, 0.2237, 1.0],
        [0.72596, 0.9647, 0.2064, 1.0],
        [0.78563, 0.93579, 0.20336, 1.0],
        [0.84133, 0.89986, 0.20926, 1.0],
        [0.89112, 0.85826, 0.2188, 1.0],
        [0.93301, 0.81236, 0.22667, 1.0],
        [0.96507, 0.76352, 0.22754, 1.0],
        [0.98549, 0.7125, 0.2165, 1.0],
        [0.99535, 0.65341, 0.19577, 1.0],
        [0.99593, 0.58703, 0.16899, 1.0],
        [0.98799, 0.51667, 0.13883, 1.0],
        [0.97234, 0.44565, 0.10797, 1.0],
        [0.94977, 0.37729, 0.07905, 1.0],
        [0.92105, 0.31489, 0.05475, 1.0],
        [0.88691, 0.26152, 0.03753, 1.0],
        [0.84662, 0.21407, 0.02487, 1.0],
        [0.79971, 0.17055, 0.0152, 1.0],
        [0.74617, 0.13098, 0.00851, 1.0],
        [0.68602, 0.09536, 0.00481, 1.0],
        [0.61923, 0.06367, 0.0041, 1.0],
        [0.54583, 0.03593, 0.00638, 1.0],
        [0.4796, 0.01583, 0.01055, 1.0],
    ]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Clear pre-existing objects
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load mesh file
bpy.ops.wm.obj_import(
    filepath=mesh_file,
    directory=str(Path(mesh_file).parent),
    files=[
        {"name": "odm_textured_model_geo.obj", "name": "odm_textured_model_geo.obj"}
    ],
    forward_axis="Y",
    up_axis="Z",
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add sun light
bpy.ops.object.light_add(type="SUN", location=(0, 0, 0))

sun_light = bpy.data.lights["Sun"]
sun_light.energy = 1.0
sun_light.use_shadow = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get all CSV files in the directory
list_csv_filepaths = list(Path(csv_dir).glob("*.csv"))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create a curve for each CSV file
for csv_i, csv_filepath in enumerate(list_csv_filepaths):
    # Read coordinates from the CSV file
    with open(csv_filepath, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        points = [(float(x), float(y), float(z) + z_offset) for x, y, z in reader]

    # Create a new curve data object
    curve_data = bpy.data.curves.new(name=f"polyline-{csv_filepath.stem}", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2

    # Create a new curve object
    curve_object = bpy.data.objects.new(
        f"polylineObject-{csv_filepath.stem}", curve_data
    )
    bpy.context.collection.objects.link(curve_object)

    # Create a polyline from the points
    polyline = curve_data.splines.new(type="POLY")
    polyline.points.add(len(points) - 1)  # Add points to the polyline
    for p_i, (x, y, z) in enumerate(points):
        polyline.points[p_i].co = (
            x,
            y,
            z,
            1,
        )  # The last value is the weight of the data point

    # Create a new material
    material = bpy.data.materials.new(name=f"material-{csv_filepath.stem}")
    material.use_nodes = True  # Enable nodes for the material

    # Set the base color of the material
    principled_bsdf = material.node_tree.nodes.get("Principled BSDF")
    principled_bsdf.inputs["Base Color"].default_value = tuple(colors_array[csv_i, :])

    # Assign the material to the curve object
    if curve_object.data.materials:
        curve_object.data.materials[0] = material
    else:
        curve_object.data.materials.append(material)

    # Set thickness
    curve_object.data.bevel_depth = 0.0015

    # Disable shadow casting for the curves
    curve_object.display.show_shadows = False


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add a camera at a specific position
camera_data = bpy.data.cameras.new(name="Camera")
camera_object = bpy.data.objects.new("CameraObject", camera_data)
bpy.context.collection.objects.link(camera_object)

# Set the camera position
camera_object.location = tuple(plane_center + camera_position_offset)

# Create a rotation that maps the camera's local z-axis to the negative plane normal
target_vector = mathutils.Vector(-plane_normal).normalized()
rot_quat = target_vector.to_track_quat("Z", "Y")  # Align Z to vector, keep Y as up

# Create a quaternion representing a 10° rotation around local Z
z_rot_quat = mathutils.Quaternion((0, 0, 1), np.deg2rad(-35))

# Apply the rotations
# To apply the rotation in z in the camera's local space, we need to apply the
# rotation to the "world" coordinate system first, and then apply the camera pose to that.
camera_object.rotation_mode = "QUATERNION"
camera_object.rotation_quaternion = rot_quat @ z_rot_quat


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Render view from the camera with default camera settings
# Set the camera as the active camera
bpy.context.scene.camera = camera_object

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.filepath = str(
    data_dir / f"orthophoto_bestfit_plane_{timestamp}.png"
)

# Set the resolution for orthophoto
bpy.context.scene.render.resolution_x = 1920  # width
bpy.context.scene.render.resolution_y = 1080  # height
bpy.context.scene.render.resolution_percentage = (
    100  # resolution percentage (100% for full resolution)
)
bpy.context.scene.render.film_transparent = True  # Enable transparent background

# Render the scene
bpy.ops.render.render(write_still=True)
