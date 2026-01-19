#!/usr/bin/env python3
"""
Camera-centric LiDAR projection.

Unlike project.py (LiDAR-centric), this script:
- Iterates over camera frames as the reference
- For each camera frame, collects LiDAR points within ±N ms time window
- Points may come from multiple LiDAR spins
- No motion compensation (±N ms is short enough)
- Uses vectorized operations for speed
"""

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import DracoPy
import cv2
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
DURATION_US = 1_000_000  # Duration in microseconds (1s)
TARGET_FPS = 30  # Output video frame rate (match camera fps)
SCALE_FACTOR = 0.5  # Scale down each camera image to reduce final video size
TIME_WINDOW_MS = 4  # ±N ms time window for selecting LiDAR points

# All 7 cameras with their layout positions
CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_rear_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
]

# =============================================================================
# Initialize dataset interface
# =============================================================================
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[2]
chunk_id = ds.get_clip_chunk(clip_id)

# Download required data
print("Downloading data...")
ds.download_chunk_features(int(chunk_id), features=ds.features.CAMERA.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.LIDAR.LIDAR_TOP_360FOV)
ds.download_chunk_features(int(chunk_id), features=ds.features.CALIBRATION.ALL)

# =============================================================================
# Load calibration data
# =============================================================================
camera_intrinsics = ds.get_clip_feature(clip_id, "camera_intrinsics")
sensor_extrinsics = ds.get_clip_feature(clip_id, "sensor_extrinsics")


# =============================================================================
# Helper functions
# =============================================================================
def get_transform_matrix(ext):
    """Build 4x4 transformation matrix from extrinsics (sensor to rig frame)."""
    quat = [ext['qx'], ext['qy'], ext['qz'], ext['qw']]
    trans = [ext['x'], ext['y'], ext['z']]
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T


def decode_lidar_timestamps(mesh, reference_timestamp):
    """Decode per-point timestamps from DracoPy mesh colors."""
    colors = np.array(mesh.colors) if mesh.colors is not None else None
    if colors is None or len(colors) == 0:
        return np.full(len(mesh.points), reference_timestamp, dtype=np.float64)

    green = colors[:, 1].astype(np.uint16)
    blue = colors[:, 2].astype(np.uint16)
    encoded_uint16 = (blue << 8) | green
    encoded_int16 = encoded_uint16.view(np.int16)
    scale = 105000.0 / 32767.0
    relative_ts = encoded_int16.astype(np.float64) * scale
    absolute_ts = reference_timestamp + relative_ts - 105000.0
    return absolute_ts


def project_lidar_to_camera_vectorized(points_lidar, T_cam_from_lidar, fw_poly, cx, cy, width, height):
    """
    Project LiDAR points to camera pixel coordinates using f-theta model.
    Fully vectorized - no point-level loops.

    Returns:
        pixels: (M, 2) array of valid pixel coordinates
        depths: (M,) array of depths for valid points
    """
    if len(points_lidar) == 0:
        return np.empty((0, 2)), np.empty(0)

    N = points_lidar.shape[0]
    points_homo = np.hstack([points_lidar, np.ones((N, 1))])
    points_cam = (T_cam_from_lidar @ points_homo.T).T[:, :3]

    x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid_depth = z_cam > 0.1

    r_xy = np.sqrt(x_cam**2 + y_cam**2)
    theta = np.arctan2(r_xy, z_cam)

    # Vectorized polynomial evaluation
    r_pixel = np.zeros_like(theta)
    theta_power = np.ones_like(theta)
    for coef in fw_poly:
        r_pixel += coef * theta_power
        theta_power *= theta

    phi = np.arctan2(y_cam, x_cam)
    u = cx + r_pixel * np.cos(phi)
    v = cy + r_pixel * np.sin(phi)

    valid_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_mask = valid_depth & valid_bounds

    pixels = np.stack([u[valid_mask], v[valid_mask]], axis=1)
    depths = z_cam[valid_mask]
    return pixels, depths


def depth_to_color_rgb(depths, d_min=0.1, d_max=80.0):
    """Convert depth values to RGB colors using turbo colormap."""
    d_normalized = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
    cmap_func = plt.colormaps.get_cmap('turbo')
    colors = cmap_func(d_normalized)[:, :3]
    return (colors * 255).astype(np.uint8)


def render_points_on_image(image, pixels, depths, point_size=2):
    """
    Render projected points on image using vectorized operations.
    Uses OpenCV's efficient drawing.
    """
    if len(pixels) == 0:
        return image.copy()

    output = image.copy()
    point_colors = depth_to_color_rgb(depths)

    # Convert to integer coordinates
    pixels_int = pixels.astype(np.int32)

    # Draw all circles - OpenCV doesn't have a fully vectorized circle draw,
    # but we can use polylines for speed or just iterate (still fast for ~100k points)
    # For best performance with many points, we draw directly on the image array
    for i in range(len(pixels_int)):
        cv2.circle(output, (pixels_int[i, 0], pixels_int[i, 1]),
                   point_size, point_colors[i].tolist(), -1)

    return output


# =============================================================================
# Prepare camera data structures
# =============================================================================
print("Loading camera parameters...")

# LiDAR extrinsics
lidar_ext = sensor_extrinsics.loc["lidar_top_360fov"]
T_rig_from_lidar = get_transform_matrix(lidar_ext)

# Store camera-specific data
camera_data = {}
for cam_name in CAMERAS:
    cam_intrinsic = camera_intrinsics.loc[cam_name]
    cam_ext = sensor_extrinsics.loc[cam_name]

    T_rig_from_cam = get_transform_matrix(cam_ext)
    T_cam_from_rig = np.linalg.inv(T_rig_from_cam)
    T_cam_from_lidar = T_cam_from_rig @ T_rig_from_lidar

    camera_data[cam_name] = {
        'reader': ds.get_clip_feature(clip_id, cam_name),
        'width': int(cam_intrinsic['width']),
        'height': int(cam_intrinsic['height']),
        'cx': cam_intrinsic['cx'],
        'cy': cam_intrinsic['cy'],
        'fw_poly': np.array([
            cam_intrinsic['fw_poly_0'],
            cam_intrinsic['fw_poly_1'],
            cam_intrinsic['fw_poly_2'],
            cam_intrinsic['fw_poly_3'],
            cam_intrinsic['fw_poly_4']
        ]),
        'T_cam_from_lidar': T_cam_from_lidar,
    }

# Get single camera dimensions
single_width = camera_data[CAMERAS[0]]['width']
single_height = camera_data[CAMERAS[0]]['height']

# Scaled dimensions
scaled_width = int(single_width * SCALE_FACTOR)
scaled_height = int(single_height * SCALE_FACTOR)

# =============================================================================
# Layout calculation
# =============================================================================
canvas_width = scaled_width * 5
canvas_height = scaled_height * 3

camera_positions = {
    "camera_front_tele_30fov":    (scaled_width * 2, 0),
    "camera_rear_left_70fov":     (0, scaled_height),
    "camera_cross_left_120fov":   (scaled_width, scaled_height),
    "camera_front_wide_120fov":   (scaled_width * 2, scaled_height),
    "camera_cross_right_120fov":  (scaled_width * 3, scaled_height),
    "camera_rear_right_70fov":    (scaled_width * 4, scaled_height),
    "camera_rear_tele_30fov":     (scaled_width * 2, scaled_height * 2),
}

# =============================================================================
# Load and preprocess all LiDAR data
# =============================================================================
print("Loading LiDAR data...")
lidar_data = ds.get_clip_feature(clip_id, "lidar_top_360fov")
lidar_df = list(lidar_data.values())[0]
print(f"Total LiDAR scans: {len(lidar_df)}")

# Preload all LiDAR points and timestamps for the duration
print("Preprocessing LiDAR point clouds...")
all_points = []
all_timestamps = []

for idx in tqdm(range(len(lidar_df)), desc="Decoding LiDAR"):
    lidar_scan = lidar_df.iloc[idx]
    reference_timestamp = lidar_scan['reference_timestamp']

    # Skip scans outside our time range (with some margin for the time window)
    if reference_timestamp > DURATION_US + TIME_WINDOW_MS * 1000:
        continue
    if reference_timestamp < -TIME_WINDOW_MS * 1000 - 105000:  # LiDAR spin can have points up to 105ms before reference
        continue

    draco_bytes = lidar_scan['draco_encoded_pointcloud']
    mesh = DracoPy.decode(draco_bytes)
    points = np.array(mesh.points)
    timestamps = decode_lidar_timestamps(mesh, reference_timestamp)

    all_points.append(points)
    all_timestamps.append(timestamps)

# Concatenate all points and timestamps
all_points = np.vstack(all_points)
all_timestamps = np.concatenate(all_timestamps)
print(f"Total LiDAR points loaded: {len(all_points)}")

# Sort by timestamp for efficient querying
sort_indices = np.argsort(all_timestamps)
all_points = all_points[sort_indices]
all_timestamps = all_timestamps[sort_indices]

# =============================================================================
# Get camera frame timestamps (use front_wide as reference for frame timing)
# =============================================================================
ref_camera = camera_data["camera_front_wide_120fov"]
camera_timestamps = ref_camera['reader'].timestamps

# Filter camera frames within duration
camera_frame_indices = np.where(camera_timestamps < DURATION_US)[0]
print(f"Processing {len(camera_frame_indices)} camera frames")

# =============================================================================
# Create video (camera-centric)
# =============================================================================
output_path = "lidar_projection_camera_centric.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (canvas_width, canvas_height))

print(f"Output video size: {canvas_width} x {canvas_height}")
print(f"Time window: ±{TIME_WINDOW_MS} ms")
print("Rendering frames (camera-centric)...")

time_window_us = TIME_WINDOW_MS * 1000  # Convert to microseconds

for frame_idx in tqdm(camera_frame_indices, desc="Rendering"):
    # Get the timestamp for this frame (from front_wide camera)
    frame_ts = camera_timestamps[frame_idx]

    # Find LiDAR points within ±N ms of this camera frame (vectorized)
    t_min = frame_ts - time_window_us
    t_max = frame_ts + time_window_us

    # Use searchsorted for O(log n) lookup
    idx_start = np.searchsorted(all_timestamps, t_min, side='left')
    idx_end = np.searchsorted(all_timestamps, t_max, side='right')

    # Get points in time window
    points_in_window = all_points[idx_start:idx_end]

    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Render each camera
    for cam_name in CAMERAS:
        cam = camera_data[cam_name]

        # Find nearest frame for this camera
        cam_timestamps_arr = cam['reader'].timestamps
        cam_frame_idx = np.argmin(np.abs(cam_timestamps_arr - frame_ts))

        # Decode camera image
        cam_image = cam['reader'].decode_images_from_frame_indices(
            np.array([cam_frame_idx])
        )[0]

        # Project LiDAR points (vectorized)
        pixels, depths = project_lidar_to_camera_vectorized(
            points_in_window,
            cam['T_cam_from_lidar'], cam['fw_poly'],
            cam['cx'], cam['cy'], cam['width'], cam['height']
        )

        # Render points on image
        rendered = render_points_on_image(cam_image, pixels, depths)

        # Scale down
        rendered_scaled = cv2.resize(rendered, (scaled_width, scaled_height),
                                      interpolation=cv2.INTER_AREA)

        # Place in canvas
        x_off, y_off = camera_positions[cam_name]
        canvas[y_off:y_off + scaled_height, x_off:x_off + scaled_width] = rendered_scaled

    # Convert RGB to BGR for OpenCV
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    video_writer.write(canvas_bgr)

video_writer.release()
print(f"Saved video to: {output_path}")
print(f"Points per frame (avg): ~{len(all_points) / len(camera_frame_indices):.0f}")
