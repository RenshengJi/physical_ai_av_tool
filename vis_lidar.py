"""
Interactive LiDAR point cloud visualization using Viser.

Features:
- Frame-by-frame navigation (Next/Previous buttons)
- Slider control for frame selection
- Auto-play with adjustable playback speed
- Color by depth or height
- World coordinate system (first frame lidar as origin)
- Accumulation mode (show frames 1~T)
- Max distance display per frame
- Motion compensation for rolling shutter correction
"""

import numpy as np
import DracoPy
import viser
import time
import threading
from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
from scipy.spatial.transform import Rotation

# =============================================================================
# Configuration
# =============================================================================
POINT_SIZE = 0.02  # Point size in meters
MAX_FRAMES = None  # Set to limit number of frames (None = all frames)

# =============================================================================
# Initialize dataset
# =============================================================================
print("Initializing dataset...")
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[1]
chunk_id = ds.get_clip_chunk(clip_id)

# Download LiDAR data and egomotion
print("Downloading LiDAR data and egomotion...")
ds.download_chunk_features(int(chunk_id), features=ds.features.LIDAR.LIDAR_TOP_360FOV)
ds.download_chunk_features(int(chunk_id), features=ds.features.CALIBRATION.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.LABELS.EGOMOTION)

# Load LiDAR data
print("Loading LiDAR data...")
lidar_data = ds.get_clip_feature(clip_id, "lidar_top_360fov")
lidar_df = list(lidar_data.values())[0]

# Load egomotion interpolator
print("Loading egomotion...")
egomotion_interp = ds.get_clip_feature(clip_id, "egomotion")

# Load sensor extrinsics
sensor_extrinsics = ds.get_clip_feature(clip_id, "sensor_extrinsics")
lidar_ext = sensor_extrinsics.loc["lidar_top_360fov"]

# Limit frames if specified
if MAX_FRAMES is not None:
    lidar_df = lidar_df.iloc[:MAX_FRAMES]

total_frames = len(lidar_df)
print(f"Total LiDAR frames: {total_frames}")


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


# LiDAR to rig transform
T_rig_from_lidar = get_transform_matrix(lidar_ext)
T_lidar_from_rig = np.linalg.inv(T_rig_from_lidar)

# Get first frame timestamp and egomotion as world origin
first_timestamp = lidar_df.iloc[0]['reference_timestamp']
first_ego = egomotion_interp(first_timestamp)
T_anchor_from_rig_first = get_egomotion_transform(first_ego)
T_rig_first_from_anchor = np.linalg.inv(T_anchor_from_rig_first)

# World frame is the first frame's lidar position
# T_world_from_lidar_first = I (identity, this is our world origin)
# For any other frame: T_world_from_lidar_t = T_lidar_first_from_lidar_t


def motion_compensate_points(points, point_timestamps, target_timestamp):
    """
    Apply motion compensation to LiDAR points (vectorized version).
    Compensates for rolling shutter effect by transforming each point
    from its capture time to the target timestamp.
    """
    N = points.shape[0]
    target_ego = egomotion_interp(target_timestamp)
    T_anchor_from_rig_target = get_egomotion_transform(target_ego)
    T_rig_target_from_anchor = np.linalg.inv(T_anchor_from_rig_target)

    points_homo = np.hstack([points, np.ones((N, 1))])

    # Quantize timestamps to reduce computation (1ms resolution)
    ts_quantized = (point_timestamps // 1000) * 1000
    unique_ts, inverse_indices = np.unique(ts_quantized, return_inverse=True)

    num_unique = len(unique_ts)
    transforms = np.zeros((num_unique, 4, 4))

    for i, ts in enumerate(unique_ts):
        try:
            capture_ego = egomotion_interp(ts)
            T_anchor_from_rig_capture = get_egomotion_transform(capture_ego)
            # Transform: lidar_capture -> rig_capture -> anchor -> rig_target -> lidar_target
            T_full = T_lidar_from_rig @ T_rig_target_from_anchor @ T_anchor_from_rig_capture @ T_rig_from_lidar
            transforms[i] = T_full
        except Exception:
            transforms[i] = np.eye(4)

    point_transforms = transforms[inverse_indices]
    compensated_homo = np.einsum('nij,nj->ni', point_transforms, points_homo)
    compensated_points = compensated_homo[:, :3]
    return compensated_points


def get_lidar_to_world_transform(timestamp):
    """Get transform from current lidar frame to world (first frame lidar)."""
    # Get current egomotion
    current_ego = egomotion_interp(timestamp)
    T_anchor_from_rig_current = get_egomotion_transform(current_ego)

    # T_world_from_lidar_current = T_lidar_first_from_lidar_current
    # = T_lidar_first_from_rig_first @ T_rig_first_from_anchor @ T_anchor_from_rig_current @ T_rig_from_lidar
    T_world_from_lidar = (
        T_lidar_from_rig @
        T_rig_first_from_anchor @
        T_anchor_from_rig_current @
        T_rig_from_lidar
    )
    return T_world_from_lidar


def decode_lidar_frame(idx, apply_motion_comp=False):
    """Decode a single LiDAR frame and return points (optionally motion compensated)."""
    scan = lidar_df.iloc[idx]
    draco_bytes = scan['draco_encoded_pointcloud']
    reference_timestamp = scan['reference_timestamp']
    mesh = DracoPy.decode(draco_bytes)
    points = np.array(mesh.points)

    if apply_motion_comp:
        point_timestamps = decode_lidar_timestamps(mesh, reference_timestamp)
        points = motion_compensate_points(points, point_timestamps, reference_timestamp)

    return points, reference_timestamp


def transform_points_to_world(points, timestamp):
    """Transform points from lidar frame to world frame."""
    T = get_lidar_to_world_transform(timestamp)
    points_homo = np.hstack([points, np.ones((len(points), 1))])
    points_world = (T @ points_homo.T).T[:, :3]
    return points_world


def depth_to_color(points, d_min=0.1, d_max=80.0):
    """Convert depth (distance from origin) to RGB colors using turbo-like colormap."""
    depths = np.linalg.norm(points[:, :2], axis=1)  # XY distance
    d_normalized = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)

    # Simple turbo-like colormap
    colors = np.zeros((len(points), 3))
    colors[:, 0] = np.clip(4 * d_normalized - 1.5, 0, 1)  # Red
    colors[:, 1] = np.clip(2 - 4 * np.abs(d_normalized - 0.5), 0, 1)  # Green
    colors[:, 2] = np.clip(1.5 - 4 * d_normalized, 0, 1)  # Blue

    return colors


def height_to_color(points, z_min=-2.0, z_max=5.0):
    """Convert height (Z) to RGB colors."""
    z = points[:, 2]
    z_normalized = np.clip((z - z_min) / (z_max - z_min + 1e-6), 0, 1)

    # Rainbow colormap based on height
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
color_mode = "depth"  # "depth", "height", or "time"
point_cloud_handle = None
updating_slider = False  # Flag to prevent recursive updates
use_world_coords = True  # Use world coordinate system
accumulate_frames = False  # Accumulate frames 0~T
use_motion_comp = True  # Use motion compensation


def update_point_cloud(frame_idx, update_slider=True):
    """Update the point cloud visualization for a given frame."""
    global point_cloud_handle, current_frame, updating_slider

    current_frame = frame_idx

    if accumulate_frames:
        # Accumulate all frames from 0 to frame_idx
        all_points = []
        all_frame_indices = []

        for i in range(frame_idx + 1):
            pts, timestamp = decode_lidar_frame(i, apply_motion_comp=use_motion_comp)

            if use_world_coords:
                pts = transform_points_to_world(pts, timestamp)

            all_points.append(pts)
            all_frame_indices.append(np.full(len(pts), i))

        points = np.vstack(all_points)
        frame_indices = np.concatenate(all_frame_indices)

        # Get colors based on mode
        if color_mode == "depth":
            colors = depth_to_color(points)
        elif color_mode == "height":
            colors = height_to_color(points)
        else:  # time
            colors = frame_to_color(frame_indices, total_frames)

        # Calculate max distance for current frame only
        current_pts, _ = decode_lidar_frame(frame_idx, apply_motion_comp=False)
        max_dist = np.linalg.norm(current_pts, axis=1).max()

    else:
        # Single frame mode
        points, timestamp = decode_lidar_frame(frame_idx, apply_motion_comp=use_motion_comp)

        # Calculate max distance before transform (use raw points)
        raw_pts, _ = decode_lidar_frame(frame_idx, apply_motion_comp=False)
        max_dist = np.linalg.norm(raw_pts, axis=1).max()

        if use_world_coords:
            points = transform_points_to_world(points, timestamp)

        # Get colors based on mode
        if color_mode == "depth":
            colors = depth_to_color(points)
        elif color_mode == "height":
            colors = height_to_color(points)
        else:  # time - single frame, use depth coloring
            colors = depth_to_color(points)

    # Update or create point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="/lidar",
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

    # Update frame info with max distance
    timestamp = lidar_df.iloc[frame_idx]['reference_timestamp']
    motion_comp_str = "MC:ON" if use_motion_comp else "MC:OFF"
    frame_info.value = f"Frame: {frame_idx}/{total_frames-1} | Time: {timestamp/1e6:.3f}s | {motion_comp_str}"
    max_dist_info.value = f"Max Distance: {max_dist:.2f} m | Points: {len(points)}"


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
        options=["depth", "height", "time"],
        initial_value="depth",
    )

    # Point size slider
    point_size_slider = server.gui.add_slider(
        "Point Size",
        min=0.005,
        max=0.1,
        step=0.005,
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

    # Motion compensation checkbox
    motion_comp_checkbox = server.gui.add_checkbox(
        "Motion Compensation",
        initial_value=True,
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


@motion_comp_checkbox.on_update
def on_motion_comp_change(event):
    global use_motion_comp
    use_motion_comp = event.target.value
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
