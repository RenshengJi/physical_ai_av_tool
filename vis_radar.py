"""
Interactive Radar point cloud visualization using Viser.

Features:
- Frame-by-frame navigation (Next/Previous buttons)
- Slider control for frame selection
- Auto-play with adjustable playback speed
- Color by velocity (blue=approaching, red=receding) or depth
- World coordinate system (first frame rig as origin)
- Accumulation mode (show frames 1~T)
- Multiple radar sensors support
"""

import numpy as np
import viser
import time
import threading
from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
from scipy.spatial.transform import Rotation

# =============================================================================
# Configuration
# =============================================================================
POINT_SIZE = 0.3  # Point size in meters (larger for radar since fewer points)
MAX_FRAMES = None  # Set to limit number of frames (None = all frames)
TIME_TOLERANCE_US = 50000  # 50ms tolerance for matching radar data to reference timestamp

# All 9 radar sensors to combine
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
# Initialize dataset
# =============================================================================
print("Initializing dataset...")
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[1]
chunk_id = ds.get_clip_chunk(clip_id)

# Download Radar data and egomotion
print("Downloading Radar data and egomotion...")
ds.download_chunk_features(int(chunk_id), features=ds.features.RADAR.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.CALIBRATION.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.LABELS.EGOMOTION)

# Load calibration data
print("Loading calibration data...")
sensor_extrinsics = ds.get_clip_feature(clip_id, "sensor_extrinsics")

# Load egomotion interpolator
print("Loading egomotion...")
egomotion_interp = ds.get_clip_feature(clip_id, "egomotion")

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


def get_egomotion_transform(egomotion_state):
    """Get 4x4 transformation matrix from egomotion state."""
    pose = egomotion_state.pose
    T = pose.as_matrix()
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


# =============================================================================
# Load radar data
# =============================================================================
print("Loading radar data...")

# Find available radar sensors matching our target list
available_radars = [idx for idx in sensor_extrinsics.index if idx.startswith('radar_')]
print(f"Available radar sensors in extrinsics: {available_radars}")

# Map base radar names to actual sensor names (they may have suffixes like _srr_0, _mrr_2, etc.)
radar_name_mapping = {}  # base_name -> actual_name
for base_name in RADARS:
    for actual_name in available_radars:
        if actual_name.startswith(base_name):
            radar_name_mapping[base_name] = actual_name
            break

print(f"Radar name mapping: {radar_name_mapping}")

# Load radar data and transforms
radar_data = {}
radar_transforms = {}  # T_rig_from_radar

for base_name, actual_name in radar_name_mapping.items():
    try:
        radar_feature = ds.get_clip_feature(clip_id, actual_name)

        # radar_feature might be a dict with dataframes
        if isinstance(radar_feature, dict):
            radar_df = list(radar_feature.values())[0]
        else:
            radar_df = radar_feature

        if radar_df is not None and len(radar_df) > 0:
            radar_data[actual_name] = radar_df

            # Get transform from radar to rig frame
            radar_ext = sensor_extrinsics.loc[actual_name]
            T_rig_from_radar = get_transform_matrix(radar_ext)
            radar_transforms[actual_name] = T_rig_from_radar

            print(f"  Loaded {actual_name}: {len(radar_df)} detections")
    except Exception as e:
        print(f"  Could not load {actual_name}: {e}")

if len(radar_data) == 0:
    print("No radar data found!")
    exit(1)

print(f"Successfully loaded {len(radar_data)} radar sensors")

# Use a reference radar to get frame timestamps (use the one with most data)
reference_radar = max(radar_data.keys(), key=lambda k: len(radar_data[k]['timestamp'].unique()))
reference_timestamps = sorted(radar_data[reference_radar]['timestamp'].unique())
print(f"Using {reference_radar} as reference with {len(reference_timestamps)} unique timestamps")

# Limit frames if specified
if MAX_FRAMES is not None:
    reference_timestamps = reference_timestamps[:MAX_FRAMES]

total_frames = len(reference_timestamps)
print(f"Total frames: {total_frames}")

# Get first frame timestamp and egomotion as world origin
first_timestamp = reference_timestamps[0]
first_ego = egomotion_interp(first_timestamp)
T_anchor_from_rig_first = get_egomotion_transform(first_ego)
T_rig_first_from_anchor = np.linalg.inv(T_anchor_from_rig_first)


# =============================================================================
# Point cloud processing functions
# =============================================================================
def get_rig_to_world_transform(timestamp):
    """Get transform from current rig frame to world (first frame rig)."""
    current_ego = egomotion_interp(timestamp)
    T_anchor_from_rig_current = get_egomotion_transform(current_ego)

    # T_world_from_rig_current = T_rig_first_from_rig_current
    T_world_from_rig = T_rig_first_from_anchor @ T_anchor_from_rig_current
    return T_world_from_rig


def decode_radar_frame(idx):
    """
    Decode a single radar frame - combines all 9 radars at the reference timestamp.
    For each radar, finds the nearest scan within TIME_TOLERANCE_US.

    Returns:
        points: Nx3 array of points in rig frame
        velocities: N array of radial velocities
        distances: N array of distances
        timestamp: the reference timestamp
    """
    target_ts = reference_timestamps[idx]

    all_points = []
    all_velocities = []
    all_distances = []

    for radar_name, radar_df in radar_data.items():
        # Find the nearest scan timestamp for this radar
        unique_scans = radar_df['timestamp'].unique()
        ts_diffs = np.abs(unique_scans - target_ts)
        min_diff_idx = np.argmin(ts_diffs)
        min_diff = ts_diffs[min_diff_idx]

        # Skip if no scan within tolerance
        if min_diff > TIME_TOLERANCE_US:
            continue

        nearest_scan_ts = unique_scans[min_diff_idx]

        # Get all detections from this scan
        scan_mask = radar_df['timestamp'] == nearest_scan_ts
        scan_detections = radar_df[scan_mask]

        if len(scan_detections) == 0:
            continue

        # Convert spherical to Cartesian in radar frame
        azimuths = scan_detections['azimuth'].values
        elevations = scan_detections['elevation'].values
        distances = scan_detections['distance'].values
        velocities = scan_detections['radial_velocity'].values

        x, y, z = radar_spherical_to_cartesian(azimuths, elevations, distances)
        points_radar = np.stack([x, y, z], axis=1)

        # Transform to rig frame
        T_rig_from_radar = radar_transforms[radar_name]
        points_homo = np.hstack([points_radar, np.ones((len(points_radar), 1))])
        points_rig = (T_rig_from_radar @ points_homo.T).T[:, :3]

        all_points.append(points_rig)
        all_velocities.append(velocities)
        all_distances.append(distances)

    if len(all_points) == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]), target_ts

    points = np.vstack(all_points)
    velocities = np.concatenate(all_velocities)
    distances = np.concatenate(all_distances)

    return points, velocities, distances, target_ts


def transform_points_to_world(points, timestamp):
    """Transform points from rig frame to world frame."""
    if len(points) == 0:
        return points
    T = get_rig_to_world_transform(timestamp)
    points_homo = np.hstack([points, np.ones((len(points), 1))])
    points_world = (T @ points_homo.T).T[:, :3]
    return points_world


def velocity_to_color(velocities, static_threshold=5.0):
    """
    Convert radial velocity values to RGB colors.
    Blue = approaching (negative), Red = receding (positive), Green = static.

    Args:
        velocities: array of radial velocities (m/s)
        static_threshold: velocities with |v| < threshold are considered static (default 1.0 m/s)
    """
    if len(velocities) == 0:
        return np.array([]).reshape(0, 3)

    colors = np.zeros((len(velocities), 3))

    # Static objects (|velocity| < threshold) -> Green
    static_mask = np.abs(velocities) < static_threshold
    colors[static_mask, 0] = 0.2   # R
    colors[static_mask, 1] = 0.8   # G
    colors[static_mask, 2] = 0.2   # B

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
            colors[neg_indices, 0] = 0.2 * (1 - intensity_neg)  # R
            colors[neg_indices, 1] = 0.2 * (1 - intensity_neg)  # G
            colors[neg_indices, 2] = 0.5 + 0.5 * intensity_neg  # B

        # Positive velocity (receding) -> Red
        pos_mask = v_normalized > 0
        if pos_mask.any():
            intensity_pos = v_normalized[pos_mask]
            moving_indices = np.where(moving_mask)[0]
            pos_indices = moving_indices[pos_mask]
            colors[pos_indices, 0] = 0.5 + 0.5 * intensity_pos  # R
            colors[pos_indices, 1] = 0.2 * (1 - intensity_pos)  # G
            colors[pos_indices, 2] = 0.2 * (1 - intensity_pos)  # B

    return colors


def depth_to_color(distances, d_min=0.1, d_max=80.0):
    """Convert depth (distance) to RGB colors using turbo-like colormap."""
    if len(distances) == 0:
        return np.array([]).reshape(0, 3)

    d_normalized = np.clip((distances - d_min) / (d_max - d_min + 1e-6), 0, 1)

    # Simple turbo-like colormap
    colors = np.zeros((len(distances), 3))
    colors[:, 0] = np.clip(4 * d_normalized - 1.5, 0, 1)  # Red
    colors[:, 1] = np.clip(2 - 4 * np.abs(d_normalized - 0.5), 0, 1)  # Green
    colors[:, 2] = np.clip(1.5 - 4 * d_normalized, 0, 1)  # Blue

    return colors


def height_to_color(points, z_min=-2.0, z_max=5.0):
    """Convert height (Z) to RGB colors."""
    if len(points) == 0:
        return np.array([]).reshape(0, 3)

    z = points[:, 2]
    z_normalized = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)

    colors = np.zeros((len(points), 3))
    colors[:, 0] = np.clip(4 * z_normalized - 1.5, 0, 1)
    colors[:, 1] = np.clip(2 - 4 * np.abs(z_normalized - 0.5), 0, 1)
    colors[:, 2] = np.clip(1.5 - 4 * z_normalized, 0, 1)

    return colors


def frame_to_color(frame_indices, total_frames):
    """Color points by their frame index (for accumulated view)."""
    t_normalized = frame_indices / (total_frames - 1 + 1e-6)

    colors = np.zeros((len(frame_indices), 3))
    colors[:, 0] = np.clip(4 * t_normalized - 1.5, 0, 1)
    colors[:, 1] = np.clip(2 - 4 * np.abs(t_normalized - 0.5), 0, 1)
    colors[:, 2] = np.clip(1.5 - 4 * t_normalized, 0, 1)

    return colors


# =============================================================================
# Viser visualization
# =============================================================================
print("Starting Viser server...")
server = viser.ViserServer()

# State variables
current_frame = 0
is_playing = False
play_thread = None
color_mode = "velocity"  # "velocity", "depth", "height", or "time"
point_cloud_handle = None
updating_slider = False  # Flag to prevent recursive updates
use_world_coords = True  # Use world coordinate system
accumulate_frames = False  # Accumulate frames 0~T


def update_point_cloud(frame_idx, update_slider=True):
    """Update the point cloud visualization for a given frame."""
    global point_cloud_handle, current_frame, updating_slider

    current_frame = frame_idx

    if accumulate_frames:
        # Accumulate all frames from 0 to frame_idx
        all_points = []
        all_velocities = []
        all_distances = []
        all_frame_indices = []

        for i in range(frame_idx + 1):
            pts, vels, dists, timestamp = decode_radar_frame(i)

            if len(pts) == 0:
                continue

            if use_world_coords:
                pts = transform_points_to_world(pts, timestamp)

            all_points.append(pts)
            all_velocities.append(vels)
            all_distances.append(dists)
            all_frame_indices.append(np.full(len(pts), i))

        if len(all_points) == 0:
            points = np.array([[0, 0, 0]])  # Dummy point
            colors = np.array([[0.5, 0.5, 0.5]])
            num_points = 0
            max_dist = 0
        else:
            points = np.vstack(all_points)
            velocities = np.concatenate(all_velocities)
            distances = np.concatenate(all_distances)
            frame_indices = np.concatenate(all_frame_indices)

            # Get colors based on mode
            if color_mode == "velocity":
                colors = velocity_to_color(velocities)
            elif color_mode == "depth":
                colors = depth_to_color(distances)
            elif color_mode == "height":
                colors = height_to_color(points)
            else:  # time
                colors = frame_to_color(frame_indices, total_frames)

            num_points = len(points)
            max_dist = distances.max() if len(distances) > 0 else 0

    else:
        # Single frame mode
        points, velocities, distances, timestamp = decode_radar_frame(frame_idx)

        if len(points) == 0:
            points = np.array([[0, 0, 0]])  # Dummy point
            colors = np.array([[0.5, 0.5, 0.5]])
            num_points = 0
            max_dist = 0
        else:
            if use_world_coords:
                points = transform_points_to_world(points, timestamp)

            # Get colors based on mode
            if color_mode == "velocity":
                colors = velocity_to_color(velocities)
            elif color_mode == "depth":
                colors = depth_to_color(distances)
            elif color_mode == "height":
                colors = height_to_color(points)
            else:  # time - single frame, use velocity coloring
                colors = velocity_to_color(velocities)

            num_points = len(points)
            max_dist = distances.max() if len(distances) > 0 else 0

    # Update or create point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="/radar",
        points=points,
        colors=colors,
        point_size=POINT_SIZE,
        point_shape="circle",
    )

    # Update slider value (with flag to prevent recursion)
    if update_slider:
        updating_slider = True
        frame_slider.value = frame_idx
        updating_slider = False

    # Update frame info
    timestamp = reference_timestamps[frame_idx]
    frame_info.value = f"Frame: {frame_idx}/{total_frames-1} | Time: {timestamp/1e6:.3f}s"
    max_dist_info.value = f"Max Distance: {max_dist:.2f} m | Points: {num_points} | Radars: {len(radar_data)}"


def play_loop():
    """Background thread for auto-play."""
    global current_frame, is_playing

    while is_playing:
        next_frame = (current_frame + 1) % total_frames
        update_point_cloud(next_frame)
        time.sleep(1.0 / playback_speed.value)


# =============================================================================
# GUI Controls
# =============================================================================
with server.gui.add_folder("Playback Controls"):
    # Frame slider
    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=total_frames - 1,
        step=1,
        initial_value=0,
    )

    # Playback speed
    playback_speed = server.gui.add_slider(
        "FPS",
        min=1,
        max=30,
        step=1,
        initial_value=10,
    )

    # Navigation buttons
    prev_button = server.gui.add_button("Previous Frame")
    next_button = server.gui.add_button("Next Frame")
    play_button = server.gui.add_button("Play")
    stop_button = server.gui.add_button("Stop")

with server.gui.add_folder("Visualization"):
    # Color mode dropdown
    color_dropdown = server.gui.add_dropdown(
        "Color Mode",
        options=["velocity", "depth", "height", "time"],
        initial_value="velocity",
    )

    # Point size slider
    point_size_slider = server.gui.add_slider(
        "Point Size",
        min=0.05,
        max=1.0,
        step=0.05,
        initial_value=POINT_SIZE,
    )

with server.gui.add_folder("Coordinate System"):
    # World coordinates checkbox
    world_coords_checkbox = server.gui.add_checkbox(
        "Use World Coordinates",
        initial_value=True,
    )

    # Accumulate frames checkbox
    accumulate_checkbox = server.gui.add_checkbox(
        "Accumulate Frames (0~T)",
        initial_value=False,
    )

with server.gui.add_folder("Info"):
    frame_info = server.gui.add_text(
        "Frame Info",
        initial_value=f"Frame: 0/{total_frames-1}",
        disabled=True,
    )
    max_dist_info = server.gui.add_text(
        "Max Distance",
        initial_value="Max Distance: -- m",
        disabled=True,
    )


# =============================================================================
# Event handlers
# =============================================================================
@frame_slider.on_update
def on_frame_slider_change(event):
    global updating_slider
    # Skip if this update was triggered by update_point_cloud
    if updating_slider:
        return
    if not is_playing:
        update_point_cloud(int(event.target.value), update_slider=False)


@prev_button.on_click
def on_prev_click(event):
    global is_playing
    is_playing = False
    new_frame = (current_frame - 1) % total_frames
    update_point_cloud(new_frame)


@next_button.on_click
def on_next_click(event):
    global is_playing
    is_playing = False
    new_frame = (current_frame + 1) % total_frames
    update_point_cloud(new_frame)


@play_button.on_click
def on_play_click(event):
    global is_playing, play_thread
    if not is_playing:
        is_playing = True
        play_thread = threading.Thread(target=play_loop, daemon=True)
        play_thread.start()


@stop_button.on_click
def on_stop_click(event):
    global is_playing
    is_playing = False


@color_dropdown.on_update
def on_color_change(event):
    global color_mode
    color_mode = event.target.value
    update_point_cloud(current_frame)


@point_size_slider.on_update
def on_point_size_change(event):
    global POINT_SIZE
    POINT_SIZE = event.target.value
    update_point_cloud(current_frame)


@world_coords_checkbox.on_update
def on_world_coords_change(event):
    global use_world_coords
    use_world_coords = event.target.value
    update_point_cloud(current_frame)


@accumulate_checkbox.on_update
def on_accumulate_change(event):
    global accumulate_frames
    accumulate_frames = event.target.value
    update_point_cloud(current_frame)


# =============================================================================
# Initial visualization
# =============================================================================
print("Loading initial frame...")
update_point_cloud(0)

# Add coordinate frame for reference
server.scene.add_frame(
    name="/origin",
    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
    position=np.array([0.0, 0.0, 0.0]),
    axes_length=2.0,
    axes_radius=0.05,
)

print(f"\nViser server running at: http://localhost:8080")
print("Open the URL in your browser to view the visualization.")
print("Press Ctrl+C to stop the server.")

# Keep the server running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down...")
