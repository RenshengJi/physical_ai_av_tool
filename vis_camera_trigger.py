#!/usr/bin/env python3
"""
绘制1秒内7个相机的触发事件时序图
横轴：时间 (ms)
纵轴：7个相机
每个触发事件用竖线表示
"""

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
DURATION_US = 1_000_000  # 1秒（微秒）

# 7个相机
CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_rear_tele_30fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
]

# 相机显示名称（简化）
CAMERA_LABELS = [
    "front_wide",
    "front_tele",
    "rear_tele",
    "cross_left",
    "cross_right",
    "rear_left",
    "rear_right",
]

# =============================================================================
# Initialize dataset interface
# =============================================================================
print("Initializing dataset interface...")
ds = PhysicalAIAVDatasetInterface(token=True)
clip_id = ds.clip_index.index[1]
chunk_id = ds.get_clip_chunk(clip_id)

# Download camera data
print("Downloading camera data...")
ds.download_chunk_features(int(chunk_id), features=ds.features.CAMERA.ALL)

# =============================================================================
# Load camera timestamps
# =============================================================================
print("Loading camera timestamps...")
camera_triggers = {}

for cam_name, cam_label in zip(CAMERAS, CAMERA_LABELS):
    reader = ds.get_clip_feature(clip_id, cam_name)
    timestamps = reader.timestamps  # 微秒
    # 筛选1秒内的触发时间
    timestamps_1s = timestamps[timestamps < DURATION_US]
    # 转换为毫秒
    timestamps_ms = timestamps_1s / 1000.0
    camera_triggers[cam_label] = timestamps_ms
    print(f"  {cam_label}: {len(timestamps_ms)} triggers in 1s")
    print(f"    timestamps (ms): {np.round(timestamps_ms, 2).tolist()}")


# =============================================================================
# Plot camera trigger events
# =============================================================================
def plot_camera_triggers(triggers, duration_ms=1000, save_path=None):
    """
    使用eventplot绘制相机触发事件时序图

    Args:
        triggers: 触发时间字典，key为相机名，value为触发时间列表(ms)
        duration_ms: 显示时长（毫秒）
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    camera_names = list(triggers.keys())
    trigger_data = [triggers[name] for name in camera_names]

    # 颜色
    colors = plt.cm.Set2(np.linspace(0, 1, len(camera_names)))

    # 使用eventplot绘制
    ax.eventplot(trigger_data, orientation='horizontal',
                 colors=colors, linewidths=1.5, linelengths=0.8)

    # 设置坐标轴
    ax.set_xlim(0, duration_ms)
    ax.set_ylim(-0.5, len(camera_names) - 0.5)

    # 设置Y轴刻度和标签
    ax.set_yticks(range(len(camera_names)))
    ax.set_yticklabels(camera_names)

    # 设置标签和标题
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Camera', fontsize=12)
    ax.set_title('Camera Trigger Events (1 second)', fontsize=14)

    # 添加网格
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # 添加统计信息
    total_triggers = sum(len(t) for t in triggers.values())
    avg_interval = duration_ms / (total_triggers / len(camera_names)) if total_triggers > 0 else 0
    ax.text(0.98, 0.02, f'Total triggers: {total_triggers} | Avg interval: {avg_interval:.1f}ms',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n绘制相机触发事件图...")
    plot_camera_triggers(
        camera_triggers,
        duration_ms=1000,
        save_path="camera_triggers.png"
    )
