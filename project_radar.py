from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import cv2
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
DURATION_US = 1_000_000  # Duration in microseconds (1s)
TARGET_FPS = 10  # Output video frame rate
SCALE_FACTOR = 0.5  # Scale down each camera image to reduce final video size

# All 7 cameras with their layout positions
CAMERAS = [
    "camera_front_wide_120fov",   # Center
    "camera_front_tele_30fov",    # Top
    "camera_rear_tele_30fov",     # Bottom
    "camera_cross_left_120fov",   # Left of center
    "camera_cross_right_120fov",  # Right of center
    "camera_rear_left_70fov",     # Far left
    "camera_rear_right_70fov",    # Far right
]

# All 9 radar sensors
RADARS = [
    "radar_front_center",
    "radar_corner_front_left",
    "radar_corner_front_right",
    "radar_side_left",
    "radar_side_right",
    "radar_corner_rear_left",
    "radar_corner_rear_right",
    "radar_rear_left",
    "radar_rear_right",
]

# =============================================================================
# Initialize dataset interface
# =============================================================================
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[1]
chunk_id = ds.get_clip_chunk(clip_id)

# Download required data
print("Downloading data...")
ds.download_chunk_features(int(chunk_id), features=ds.features.CAMERA.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.RADAR.ALL)
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


def radar_spherical_to_cartesian(azimuth, elevation, distance):
    """
    Convert radar spherical coordinates to Cartesian coordinates in radar frame.

    Args:
        azimuth: Horizontal angle (radians)
        elevation: Vertical angle (radians)
        distance: Range to target (meters)

    Returns:
        x, y, z coordinates in radar frame (meters)
    """
    # Standard spherical to Cartesian conversion
    # x: forward, y: left, z: up (following vehicle coordinate convention)
    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    return x, y, z


def project_radar_to_camera(points_radar, T_cam_from_radar, fw_poly, cx, cy, width, height):
    """Project radar points to camera pixel coordinates using f-theta model."""
    N = points_radar.shape[0]
    if N == 0:
        return np.array([]).reshape(0, 2), np.array([], dtype=bool), np.array([])

    points_homo = np.hstack([points_radar, np.ones((N, 1))])
    points_cam = (T_cam_from_radar @ points_homo.T).T[:, :3]

    x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid_depth = z_cam > 0.1

    r_xy = np.sqrt(x_cam**2 + y_cam**2)
    theta = np.arctan2(r_xy, z_cam)

    r_pixel = np.zeros_like(theta)
    for i, coef in enumerate(fw_poly):
        r_pixel += coef * (theta ** i)

    phi = np.arctan2(y_cam, x_cam)
    u = cx + r_pixel * np.cos(phi)
    v = cy + r_pixel * np.sin(phi)

    valid_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_mask = valid_depth & valid_bounds

    pixels = np.stack([u[valid_mask], v[valid_mask]], axis=1)
    depths = z_cam[valid_mask]
    return pixels, valid_mask, depths


def velocity_to_color_rgb(velocities, static_threshold=5.0):
    """
    Convert radial velocity values to RGB colors.
    Blue = approaching (negative), Red = receding (positive), Green = static.
    Uses pure RGB colors for maximum contrast.

    Args:
        velocities: array of radial velocities (m/s)
        static_threshold: velocities with |v| < threshold are considered static (default 1.0 m/s)
    """
    if len(velocities) == 0:
        return np.array([]).reshape(0, 3).astype(np.uint8)

    colors = np.zeros((len(velocities), 3), dtype=np.uint8)

    # Static objects (|velocity| < threshold) -> Green
    static_mask = np.abs(velocities) < static_threshold
    colors[static_mask, 0] = 50   # R
    colors[static_mask, 1] = 200  # G
    colors[static_mask, 2] = 50   # B

    # Moving objects
    moving_mask = ~static_mask
    if moving_mask.any():
        moving_velocities = velocities[moving_mask]
        v_abs_max = max(np.abs(moving_velocities).max(), static_threshold + 0.1)

        # Normalize moving velocities to [-1, 1]
        v_normalized = np.clip(moving_velocities / (v_abs_max + 1e-6), -1, 1)

        # Negative velocity (approaching) -> Blue
        neg_mask = v_normalized < 0
        if neg_mask.any():
            intensity_neg = np.abs(v_normalized[neg_mask])
            moving_indices = np.where(moving_mask)[0]
            neg_indices = moving_indices[neg_mask]
            colors[neg_indices, 0] = (50 * (1 - intensity_neg)).astype(np.uint8)   # R
            colors[neg_indices, 1] = (50 * (1 - intensity_neg)).astype(np.uint8)   # G
            colors[neg_indices, 2] = (255 * (0.5 + 0.5 * intensity_neg)).astype(np.uint8)  # B

        # Positive velocity (receding) -> Red
        pos_mask = v_normalized > 0
        if pos_mask.any():
            intensity_pos = v_normalized[pos_mask]
            moving_indices = np.where(moving_mask)[0]
            pos_indices = moving_indices[pos_mask]
            colors[pos_indices, 0] = (255 * (0.5 + 0.5 * intensity_pos)).astype(np.uint8)  # R
            colors[pos_indices, 1] = (50 * (1 - intensity_pos)).astype(np.uint8)   # G
            colors[pos_indices, 2] = (50 * (1 - intensity_pos)).astype(np.uint8)   # B

    return colors


def depth_to_color_rgb(depths, d_min=0.1, d_max=80.0):
    """Convert depth values to RGB colors using turbo colormap."""
    d_normalized = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
    cmap_func = plt.colormaps.get_cmap('turbo')
    colors = cmap_func(d_normalized)[:, :3]
    return (colors * 255).astype(np.uint8)


def render_single_camera(camera_image, all_radar_points, all_velocities,
                         camera_transforms, fw_poly, cx, cy, width, height,
                         color_by='velocity'):
    """
    Render radar projection on a single camera image.

    Args:
        camera_image: The camera image (H, W, 3)
        all_radar_points: Dict of {radar_name: points_in_radar_frame}
        all_velocities: Dict of {radar_name: radial_velocities}
        camera_transforms: Dict of {radar_name: T_cam_from_radar}
        fw_poly: f-theta polynomial coefficients
        cx, cy: Camera principal point
        width, height: Image dimensions
        color_by: 'velocity' or 'depth'
    """
    output = camera_image.copy()

    all_pixels = []
    all_depths = []
    all_vels = []

    for radar_name, points in all_radar_points.items():
        if len(points) == 0:
            continue

        T_cam_from_radar = camera_transforms.get(radar_name)
        if T_cam_from_radar is None:
            continue

        velocities = all_velocities.get(radar_name)
        if velocities is None or len(velocities) != len(points):
            continue

        pixels, valid_mask, depths = project_radar_to_camera(
            points, T_cam_from_radar, fw_poly, cx, cy, width, height
        )

        if len(pixels) > 0:
            all_pixels.append(pixels)
            all_depths.append(depths)
            all_vels.append(velocities[valid_mask])

    if len(all_pixels) == 0:
        return output

    all_pixels = np.vstack(all_pixels)
    all_depths = np.concatenate(all_depths)
    all_vels = np.concatenate(all_vels)

    # Color based on velocity or depth
    if color_by == 'velocity':
        point_colors = velocity_to_color_rgb(all_vels)
    else:
        point_colors = depth_to_color_rgb(all_depths)

    # Draw points (larger circles for radar since fewer points)
    for (u, v), color in zip(all_pixels.astype(int), point_colors):
        cv2.circle(output, (u, v), 8, color.tolist(), -1)  # Filled circle with color

    return output


# =============================================================================
# Prepare camera data structures
# =============================================================================
print("Loading camera parameters...")

# Store camera-specific data
camera_data = {}
for cam_name in CAMERAS:
    cam_intrinsic = camera_intrinsics.loc[cam_name]
    cam_ext = sensor_extrinsics.loc[cam_name]

    T_rig_from_cam = get_transform_matrix(cam_ext)
    T_cam_from_rig = np.linalg.inv(T_rig_from_cam)

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
        'T_cam_from_rig': T_cam_from_rig,
    }

# =============================================================================
# Prepare radar data structures
# =============================================================================
print("Loading radar data...")

# Get available radars for this clip
radar_data = {}
radar_transforms = {}  # T_cam_from_radar for each camera-radar pair

# Find actual radar names in extrinsics (they have suffixes like _srr_0, _mrr_2, etc.)
available_radars = [idx for idx in sensor_extrinsics.index if idx.startswith('radar_')]
print(f"Available radar sensors: {available_radars}")

for radar_name in available_radars:
    try:
        radar_feature = ds.get_clip_feature(clip_id, radar_name)

        # radar_feature might be a dict with dataframes, similar to lidar
        if isinstance(radar_feature, dict):
            radar_df = list(radar_feature.values())[0]
        else:
            radar_df = radar_feature

        if radar_df is not None and len(radar_df) > 0:
            radar_data[radar_name] = radar_df

            # Print columns for debugging
            if len(radar_data) == 1:  # Only print once
                print(f"  Radar data columns: {radar_df.columns.tolist()}")

            # Compute transforms from this radar to each camera
            radar_ext = sensor_extrinsics.loc[radar_name]
            T_rig_from_radar = get_transform_matrix(radar_ext)

            radar_transforms[radar_name] = {}
            for cam_name in CAMERAS:
                T_cam_from_rig = camera_data[cam_name]['T_cam_from_rig']
                T_cam_from_radar = T_cam_from_rig @ T_rig_from_radar
                radar_transforms[radar_name][cam_name] = T_cam_from_radar

            print(f"  Loaded {radar_name}: {len(radar_df)} detections")
    except Exception as e:
        print(f"  Could not load {radar_name}: {e}")

# Get single camera dimensions (assuming all cameras have same resolution)
single_width = camera_data[CAMERAS[0]]['width']
single_height = camera_data[CAMERAS[0]]['height']

# Scaled dimensions
scaled_width = int(single_width * SCALE_FACTOR)
scaled_height = int(single_height * SCALE_FACTOR)

# =============================================================================
# Layout calculation (same as project.py)
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
# Get reference timestamps from camera
# =============================================================================
camera_timestamps = camera_data["camera_front_wide_120fov"]['reader'].timestamps
print(f"Total camera frames: {len(camera_timestamps)}")

# Filter timestamps within duration
camera_indices = np.where(camera_timestamps < DURATION_US)[0]
print(f"Processing {len(camera_indices)} camera frames")

# =============================================================================
# Create video
# =============================================================================
output_path = "radar_projection_7cameras.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (canvas_width, canvas_height))

print(f"Output video size: {canvas_width} x {canvas_height}")
print("Rendering frames with radar projection on all 7 cameras...")

for cam_idx in tqdm(camera_indices):
    target_ts = camera_timestamps[cam_idx]

    # Collect radar points from all radars at this timestamp
    all_radar_points = {}
    all_velocities = {}

    for radar_name, radar_df in radar_data.items():
        # Find radar detections near this camera timestamp
        # Radar data is stored per-detection, so we need to find all detections
        # with timestamp close to target_ts
        ts_diff = np.abs(radar_df['timestamp'].values - target_ts)

        # Get unique scan timestamps
        unique_scans = radar_df['timestamp'].unique()
        nearest_scan_ts = unique_scans[np.argmin(np.abs(unique_scans - target_ts))]

        # Get all detections from this scan
        scan_mask = radar_df['timestamp'] == nearest_scan_ts
        scan_detections = radar_df[scan_mask]

        if len(scan_detections) == 0:
            all_radar_points[radar_name] = np.array([]).reshape(0, 3)
            all_velocities[radar_name] = np.array([])
            continue

        # Convert spherical to Cartesian coordinates
        azimuths = scan_detections['azimuth'].values
        elevations = scan_detections['elevation'].values
        distances = scan_detections['distance'].values
        velocities = scan_detections['radial_velocity'].values

        x, y, z = radar_spherical_to_cartesian(azimuths, elevations, distances)
        points = np.stack([x, y, z], axis=1)

        all_radar_points[radar_name] = points
        all_velocities[radar_name] = velocities

    # Create canvas (black background)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Render each camera
    for cam_name in CAMERAS:
        cam = camera_data[cam_name]

        # Find nearest frame for this camera
        cam_timestamps = cam['reader'].timestamps
        cam_frame_idx = np.argmin(np.abs(cam_timestamps - target_ts))

        # Decode camera image
        cam_image = cam['reader'].decode_images_from_frame_indices(
            np.array([cam_frame_idx])
        )[0]

        # Get transforms from each radar to this camera
        camera_radar_transforms = {}
        for radar_name in radar_data.keys():
            if radar_name in radar_transforms:
                camera_radar_transforms[radar_name] = radar_transforms[radar_name][cam_name]

        # Render radar projection
        rendered = render_single_camera(
            cam_image, all_radar_points, all_velocities,
            camera_radar_transforms, cam['fw_poly'],
            cam['cx'], cam['cy'], cam['width'], cam['height'],
            color_by='depth'  # Color by depth (turbo colormap)
        )

        # Scale down the rendered image
        rendered_scaled = cv2.resize(rendered, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        # Place in canvas
        x_off, y_off = camera_positions[cam_name]
        canvas[y_off:y_off + scaled_height, x_off:x_off + scaled_width] = rendered_scaled

    # Convert RGB to BGR for OpenCV
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    video_writer.write(canvas_bgr)

video_writer.release()
print(f"Saved video to: {output_path}")
