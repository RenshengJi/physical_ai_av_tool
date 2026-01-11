from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import DracoPy
import cv2
from tqdm import tqdm

# Initialize dataset interface
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[1]  # First clip in the dataset
chunk_id = ds.get_clip_chunk(clip_id)

# Download required data
ds.download_chunk_features(int(chunk_id), features=ds.features.CAMERA.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.LIDAR.LIDAR_TOP_360FOV)
ds.download_chunk_features(int(chunk_id), features=ds.features.CALIBRATION.ALL)
ds.download_chunk_features(int(chunk_id), features=ds.features.LABELS.EGOMOTION)

# =============================================================================
# Load calibration data
# =============================================================================
camera_intrinsics = ds.get_clip_feature(clip_id, "camera_intrinsics")
sensor_extrinsics = ds.get_clip_feature(clip_id, "sensor_extrinsics")

# Choose which camera to project onto
# Options:
#   - camera_front_wide_120fov   (前视广角)
#   - camera_front_tele_30fov    (前视长焦)
#   - camera_cross_left_120fov   (左侧广角)
#   - camera_cross_right_120fov  (右侧广角)
#   - camera_rear_left_70fov     (后左)
#   - camera_rear_right_70fov    (后右)
#   - camera_rear_tele_30fov     (后视长焦)
camera_name = "camera_rear_tele_30fov"

# Get camera intrinsics for selected camera
cam_intrinsic = camera_intrinsics.loc[camera_name]
width = int(cam_intrinsic['width'])
height = int(cam_intrinsic['height'])
cx, cy = cam_intrinsic['cx'], cam_intrinsic['cy']
# f-theta forward polynomial coefficients (for 3D -> pixel projection)
fw_poly = np.array([
    cam_intrinsic['fw_poly_0'],
    cam_intrinsic['fw_poly_1'],
    cam_intrinsic['fw_poly_2'],
    cam_intrinsic['fw_poly_3'],
    cam_intrinsic['fw_poly_4']
])

# Get extrinsics for camera and lidar
cam_ext = sensor_extrinsics.loc[camera_name]
lidar_ext = sensor_extrinsics.loc["lidar_top_360fov"]

def get_transform_matrix(ext):
    """Build 4x4 transformation matrix from extrinsics (sensor to rig frame)."""
    quat = [ext['qx'], ext['qy'], ext['qz'], ext['qw']]
    trans = [ext['x'], ext['y'], ext['z']]
    R = Rotation.from_quat(quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T

# Transform matrices: sensor frame -> rig frame
T_rig_from_cam = get_transform_matrix(cam_ext)
T_rig_from_lidar = get_transform_matrix(lidar_ext)

# Compute lidar -> camera transform: T_cam_from_lidar = T_cam_from_rig @ T_rig_from_lidar
T_cam_from_rig = np.linalg.inv(T_rig_from_cam)
T_cam_from_lidar = T_cam_from_rig @ T_rig_from_lidar

# =============================================================================
# Load LiDAR and camera data
# =============================================================================
lidar_data = ds.get_clip_feature(clip_id, "lidar_top_360fov")
camera_reader = ds.get_clip_feature(clip_id, camera_name)
egomotion_interp = ds.get_clip_feature(clip_id, "egomotion")  # Egomotion interpolator

# Get LiDAR DataFrame
lidar_df = list(lidar_data.values())[0]
print(f"LiDAR DataFrame columns: {lidar_df.columns.tolist()}")
print(f"Total LiDAR scans: {len(lidar_df)}")

# Get camera timestamps
camera_timestamps = camera_reader.timestamps  # in microseconds
print(f"Total camera frames: {len(camera_timestamps)}")

# =============================================================================
# Project LiDAR points to camera
# =============================================================================
def decode_lidar_timestamps(mesh, reference_timestamp):
    """
    Decode per-point timestamps from DracoPy mesh colors.

    According to wiki: timestamps are encoded in Green + Blue channels.
    16-bit timestamp = (Blue << 8) | Green, interpreted as signed int16.
    Scaled to range [-105000, +105000] microseconds relative to reference_timestamp.
    """
    colors = np.array(mesh.colors) if mesh.colors is not None else None
    if colors is None or len(colors) == 0:
        # Fallback: all points have reference timestamp
        return np.full(len(mesh.points), reference_timestamp, dtype=np.float64)

    # Decode relative timestamp from Green and Blue channels
    green = colors[:, 1].astype(np.uint16)
    blue = colors[:, 2].astype(np.uint16)

    # Combine as uint16, then reinterpret as signed int16
    encoded_uint16 = (blue << 8) | green
    encoded_int16 = encoded_uint16.view(np.int16)  # Reinterpret as signed [-32768, 32767]

    # Scale from [-32767, 32767] to [-105000, +105000] microseconds
    scale = 105000.0 / 32767.0
    relative_ts = encoded_int16.astype(np.float64) * scale - 105000.0

    absolute_ts = reference_timestamp + relative_ts  # + 10000000  # for test

    return absolute_ts

def get_egomotion_transform(egomotion_state):
    """Get 4x4 transformation matrix from egomotion state (anchor -> rig at time t)."""
    pose = egomotion_state.pose
    # pose.as_matrix() returns 4x4 transformation matrix directly
    T = pose.as_matrix()
    return T

def motion_compensate_points(points, point_timestamps, target_timestamp,
                              egomotion_interp, T_rig_from_lidar):
    """
    Apply motion compensation to LiDAR points (vectorized version).

    Transform points from their capture time to target time (camera frame time).

    Note: egomotion_interp expects timestamps in MICROSECONDS.
    """
    N = points.shape[0]

    # Get target pose (anchor -> rig at target time)
    # egomotion_interp expects microseconds
    target_ego = egomotion_interp(target_timestamp)
    T_anchor_from_rig_target = get_egomotion_transform(target_ego)
    T_rig_target_from_anchor = np.linalg.inv(T_anchor_from_rig_target)

    # Transform points to homogeneous coordinates in lidar frame
    points_homo = np.hstack([points, np.ones((N, 1))])  # (N, 4)

    # Quantize timestamps to reduce unique values (group by 1ms bins)
    # This dramatically reduces the number of unique poses to compute
    ts_quantized = (point_timestamps // 1000) * 1000  # Round to nearest ms
    unique_ts, inverse_indices = np.unique(ts_quantized, return_inverse=True)

    # Pre-compute all unique transformations
    T_lidar_from_rig = np.linalg.inv(T_rig_from_lidar)

    # Build transformation matrices for all unique timestamps
    num_unique = len(unique_ts)
    transforms = np.zeros((num_unique, 4, 4))

    for i, ts in enumerate(unique_ts):
        try:
            # egomotion_interp expects microseconds
            capture_ego = egomotion_interp(ts)
            T_anchor_from_rig_capture = get_egomotion_transform(capture_ego)
            # Full transform: lidar_capture -> rig_capture -> anchor -> rig_target -> lidar_target
            T_full = T_lidar_from_rig @ T_rig_target_from_anchor @ T_anchor_from_rig_capture @ T_rig_from_lidar
            transforms[i] = T_full
        except Exception as e:
            print(f"Warning: egomotion interpolation failed for ts={ts} us: {e}")
            transforms[i] = np.eye(4)  # Identity if interpolation fails

    # Apply transformations using advanced indexing (vectorized)
    # Get the transform for each point based on its timestamp bin
    point_transforms = transforms[inverse_indices]  # (N, 4, 4)

    # Batch matrix-vector multiplication: (N, 4, 4) @ (N, 4, 1) -> (N, 4, 1)
    compensated_homo = np.einsum('nij,nj->ni', point_transforms, points_homo)
    compensated_points = compensated_homo[:, :3]

    return compensated_points

def project_lidar_to_camera(points_lidar, T_cam_from_lidar, fw_poly, cx, cy, width, height):
    """
    Project LiDAR points to camera pixel coordinates using f-theta model.
    """
    N = points_lidar.shape[0]
    points_homo = np.hstack([points_lidar, np.ones((N, 1))])
    points_cam = (T_cam_from_lidar @ points_homo.T).T[:, :3]

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

def depth_to_color_rgb(depths, d_min=0.1, d_max=80.0):
    """Convert depth values to RGB colors using turbo colormap."""
    d_normalized = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
    cmap_func = plt.cm.get_cmap('turbo')
    colors = cmap_func(d_normalized)[:, :3]  # RGB, drop alpha
    return (colors * 255).astype(np.uint8)

def find_nearest_lidar_scan(camera_timestamp_us, lidar_df):
    """Find the LiDAR scan closest to the camera timestamp."""
    lidar_timestamps = lidar_df['reference_timestamp'].values
    idx = np.argmin(np.abs(lidar_timestamps - camera_timestamp_us))
    return idx

def render_frame(camera_image, lidar_scan, camera_timestamp, T_cam_from_lidar,
                  T_rig_from_lidar, egomotion_interp, fw_poly, cx, cy, width, height,
                  use_motion_compensation=True):
    """Render a single frame with LiDAR projection overlay and motion compensation."""
    # Decode LiDAR point cloud
    draco_bytes = lidar_scan['draco_encoded_pointcloud']
    reference_timestamp = lidar_scan['reference_timestamp']
    mesh = DracoPy.decode(draco_bytes)
    points = np.array(mesh.points)

    # Apply motion compensation if enabled
    if use_motion_compensation:
        point_timestamps = decode_lidar_timestamps(mesh, reference_timestamp)
        points = motion_compensate_points(
            points, point_timestamps, camera_timestamp,
            egomotion_interp, T_rig_from_lidar
        )

    # Project to camera
    pixels, valid_mask, depths = project_lidar_to_camera(
        points, T_cam_from_lidar, fw_poly, cx, cy, width, height
    )

    if len(pixels) == 0:
        return camera_image.copy()

    # Create output image (copy of camera image)
    output = camera_image.copy()

    # Get colors based on depth
    point_colors = depth_to_color_rgb(depths)

    # Draw points on image
    for (u, v), color in zip(pixels.astype(int), point_colors):
        cv2.circle(output, (u, v), 1, color.tolist(), -1)

    return output

# =============================================================================
# Process first 5 seconds and create video
# =============================================================================
# Configuration
DURATION_US = 1_000_000  # Duration in microseconds (1s = 1_000_000 us)
TARGET_FPS = 30  # Output video frame rate
USE_MOTION_COMPENSATION = True  # Set to False to disable motion compensation

# Filter LiDAR scans within first 5 seconds
lidar_timestamps = lidar_df['reference_timestamp'].values
lidar_indices = np.where(lidar_timestamps < DURATION_US)[0]
print(f"Processing {len(lidar_indices)} LiDAR scans (first 5 seconds at 10Hz)")

# Setup video writer
output_path = "lidar_projection_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height))

print("Rendering frames with LiDAR projection...")
for lidar_idx in tqdm(lidar_indices):
    lidar_scan = lidar_df.iloc[lidar_idx]
    lidar_ts = lidar_scan['reference_timestamp']

    # Find nearest camera frame for this LiDAR scan
    camera_frame_idx = np.argmin(np.abs(camera_timestamps - lidar_ts))
    camera_ts = camera_timestamps[camera_frame_idx]

    # Decode single camera frame
    camera_image = camera_reader.decode_images_from_frame_indices(
        np.array([camera_frame_idx])
    )[0]

    # Render frame with LiDAR overlay
    rendered_frame = render_frame(
        camera_image, lidar_scan, camera_ts, T_cam_from_lidar,
        T_rig_from_lidar, egomotion_interp, fw_poly, cx, cy, width, height,
        use_motion_compensation=USE_MOTION_COMPENSATION
    )

    # Convert RGB to BGR for OpenCV
    rendered_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
    video_writer.write(rendered_bgr)

video_writer.release()
print(f"Saved video to: {output_path}")
