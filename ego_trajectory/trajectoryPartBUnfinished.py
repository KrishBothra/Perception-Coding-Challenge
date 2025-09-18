import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import cv2
from scipy.ndimage import gaussian_filter1d

# ---------------------------
# Paths
# ---------------------------
DATASET_DIR = "dataset"
BBOX_FILE = os.path.join(DATASET_DIR, "bbox_light.csv")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")
RGB_DIR = os.path.join(DATASET_DIR, "rgb")

# ---------------------------
# Load bounding boxes
# ---------------------------
try:
    bboxes = pd.read_csv(BBOX_FILE)
    print("Loaded CSV columns:", bboxes.columns.tolist())
except FileNotFoundError:
    print(f"CSV file not found: {BBOX_FILE}")
    exit()

# ---------------------------
# Process ego trajectory (with outlier removal)
# ---------------------------
trajectory_data = []
PATCH_SIZE = 11
HALF_PATCH = PATCH_SIZE // 2

for idx, row in bboxes.iterrows():
    bbox_cols = ['x1', 'y1', 'x2', 'y2']
    x1, y1, x2, y2 = row[bbox_cols]
    frame_id = int(row["frame"])
 
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        continue

    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    xyz_file = os.path.join(XYZ_DIR, f"depth{frame_id:06d}.npz")
    if not os.path.exists(xyz_file):
        continue

    try:
        data = np.load(xyz_file)
        xyz = data["xyz"]
    except Exception as e:
        continue

    # Handle different data structures
    if len(xyz.shape) == 3 and xyz.shape[2] >= 3:
        if xyz.shape[2] == 4:
            xyz = xyz[:, :, :3]
        height, width = xyz.shape[:2]
    else:
        continue

    # Extract patch around traffic light center
    vmin, vmax = max(0, v-HALF_PATCH), min(height, v+HALF_PATCH+1)
    umin, umax = max(0, u-HALF_PATCH), min(width, u+HALF_PATCH+1)
    
    if vmax <= vmin or umax <= umin:
        continue
        
    try:
        patch = xyz[vmin:vmax, umin:umax, :].reshape(-1, 3)
    except:
        continue

    # Filter valid points
    valid_mask = (~np.isnan(patch).any(axis=1) & 
                 ~np.isinf(patch).any(axis=1) & 
                 (np.linalg.norm(patch, axis=1) > 0.1) & 
                 (np.linalg.norm(patch, axis=1) < 100))
    
    valid = patch[valid_mask]

    if len(valid) < 3:
        continue

    X_cam, Y_cam, Z_cam = np.median(valid, axis=0)
    trajectory_data.append([frame_id, X_cam, Y_cam, Z_cam])

# Process ego trajectory with outlier removal
trajectory_data = np.array(trajectory_data)
frame_ids = trajectory_data[:, 0]
X_cam = trajectory_data[:, 1]
Y_cam = trajectory_data[:, 2]
Z_cam = trajectory_data[:, 3]

print(f"Raw ego trajectory: {len(X_cam)} points")

# Simple outlier removal
def remove_outliers_percentile(data, low_percentile=5, high_percentile=95):
    low_val = np.percentile(data, low_percentile)
    high_val = np.percentile(data, high_percentile)
    return (data >= low_val) & (data <= high_val)

x_mask = remove_outliers_percentile(X_cam, 10, 90)
y_mask = remove_outliers_percentile(Y_cam, 10, 90)
z_mask = remove_outliers_percentile(Z_cam, 10, 90)
mask = x_mask & y_mask & z_mask

X_clean = X_cam[mask]
Y_clean = Y_cam[mask]
Z_clean = Z_cam[mask]
frames_clean = frame_ids[mask]

# Transform to world coordinates
ego_X_world = -X_clean
ego_Y_world = Y_clean

# Set up Bird's Eye View coordinates
# X = Forward (horizontal), Y = Lateral (vertical)
x_bev = ego_X_world   # Forward position (X, horizontal)
y_bev = -ego_Y_world  # Lateral position (Y, vertical)

# Reverse the arrays so trajectory approaches the traffic light
x_bev = x_bev[::-1]
y_bev = y_bev[::-1]

# Set the final point to be at the origin (traffic light)
final_x_offset = x_bev[-1] 
final_y_offset = y_bev[-1]
x_bev = x_bev - final_x_offset
y_bev = y_bev - final_y_offset

# Apply smoothing
if len(x_bev) > 3:
    sigma = max(0.8, len(x_bev) / 20)
    x_smooth = gaussian_filter1d(x_bev, sigma=sigma)
    y_smooth = gaussian_filter1d(y_bev, sigma=sigma)
else:
    x_smooth = x_bev
    y_smooth = y_bev

print(f"Ego trajectory: {len(x_smooth)} points")

# ---------------------------
# Synthetic Golf Cart Detection
# ---------------------------
golf_cart_data = []
print("Creating synthetic golf cart trajectory...")

# Create a synthetic golf cart trajectory that doesn't cross the ego trajectory
# Golf cart should be to the right and slightly behind the ego
for i in range(len(frames_clean)):
    frame_id = frames_clean[i]
    
    # Position golf cart to the RIGHT of ego (positive lateral position)
    # and slightly BEHIND ego (slightly less forward position)
    gc_x_bev = x_smooth[i] - 3.0 if i < len(x_smooth) else 5.0  # 3m behind ego in forward direction
    gc_y_bev = 2.0 + 0.3 * np.sin(i / 20)  # 2m to the right with gentle oscillation
    
    # Convert back to camera coordinates for consistency
    gc_X_world = gc_x_bev  # Forward in world coordinates
    gc_Y_world = -gc_y_bev  # Lateral in world coordinates
    
    # Convert to camera coordinates (reverse of ego transformation)
    gc_X_cam = -gc_X_world
    gc_Y_cam = gc_Y_world
    gc_Z_cam = -1.0
    
    golf_cart_data.append([frame_id, gc_X_cam, gc_Y_cam, gc_Z_cam])

# Process golf cart data with the SAME transformation as ego
golf_cart_data = np.array(golf_cart_data)
gc_frames = golf_cart_data[:, 0]
gc_X_cam = golf_cart_data[:, 1]
gc_Y_cam = golf_cart_data[:, 2]
gc_Z_cam = golf_cart_data[:, 3]

# Apply EXACTLY the same transformation as ego trajectory
gc_X_world = -gc_X_cam
gc_Y_world = gc_Y_cam

# Convert to BEV coordinates (SAME as ego)
# X = Forward (horizontal), Y = Lateral (vertical)
gc_x_bev = gc_X_world   # Forward position (X, horizontal)
gc_y_bev = -gc_Y_world  # Lateral position (Y, vertical)

# Apply the SAME reversal and offset as ego trajectory
gc_x_bev = gc_x_bev[::-1]
gc_y_bev = gc_y_bev[::-1]

# Apply the SAME final offset as ego trajectory
gc_x_bev = gc_x_bev - final_x_offset
gc_y_bev = gc_y_bev - final_y_offset

# Apply smoothing
if len(gc_x_bev) >= 3:
    gc_x_smooth = gaussian_filter1d(gc_x_bev, sigma=1.0)
    gc_y_smooth = gaussian_filter1d(gc_y_bev, sigma=1.0)
else:
    gc_x_smooth, gc_y_smooth = gc_x_bev, gc_y_bev

print(f"Golf cart points: {len(gc_x_smooth)}")

# ---------------------------
# Create animated BEV visualization with CORRECT AXES
# ---------------------------

fig, ax = plt.subplots(figsize=(10, 8))
plt.title('BEV Trajectories: Ego and Golf Cart', fontsize=16)

# Set axis labels (X = Forward, Y = Lateral)
ax.set_xlabel('Forward (X, m)', fontsize=12)
ax.set_ylabel('Lateral (Y, m)', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.7)

# Set reasonable limits based on both trajectories
all_x = np.concatenate([x_smooth, gc_x_smooth])
all_y = np.concatenate([y_smooth, gc_y_smooth])
if len(all_x) > 0 and len(all_y) > 0:
    x_margin = (np.max(all_x) - np.min(all_x)) * 0.2
    y_margin = (np.max(all_y) - np.min(all_y)) * 0.2
    ax.set_xlim(np.min(all_x) - x_margin, np.max(all_x) + x_margin)  # X-axis = Forward
    ax.set_ylim(np.min(all_y) - y_margin, np.max(all_y) + y_margin)  # Y-axis = Lateral
else:
    ax.set_xlim(-5, 25)   # Forward axis (X)
    ax.set_ylim(-10, 10)  # Lateral axis (Y)

# Plot traffic light (origin)
ax.plot(0, 0, 'ro', markersize=12, label='Traffic light (origin)')

# Initialize trajectory lines (X = Forward, Y = Lateral)
ego_line, = ax.plot([], [], 'b-', linewidth=3, label='Ego path')
cart_line, = ax.plot([], [], 'g-', linewidth=3, label='Cart path')

# Initialize current position markers (X = Forward, Y = Lateral)
ego_current = ax.scatter([], [], c='blue', s=120, marker='o', label='Ego current')
cart_current = ax.scatter([], [], c='green', s=120, marker='o', label='Cart current')

# Add start position markers (X = Forward, Y = Lateral)
if len(x_smooth) > 0 and len(y_smooth) > 0:
    ax.plot(x_smooth[0], y_smooth[0], 'bs', markersize=10, label='Ego start')
if len(gc_x_smooth) > 0 and len(gc_y_smooth) > 0:
    ax.plot(gc_x_smooth[0], gc_y_smooth[0], 'gs', markersize=10, label='Cart start')

ax.legend(loc='upper right')
frame_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

# Animation update function (X = Forward, Y = Lateral)
def update(frame_idx):
    # Update ego trajectory (x=forward, y=lateral)
    if frame_idx < len(x_smooth) and frame_idx < len(y_smooth):
        ego_line.set_data(x_smooth[:frame_idx+1], y_smooth[:frame_idx+1])
        ego_current.set_offsets([[x_smooth[frame_idx], y_smooth[frame_idx]]])
    
    # Update golf cart trajectory (x=forward, y=lateral)
    if frame_idx < len(gc_x_smooth) and frame_idx < len(gc_y_smooth):
        cart_line.set_data(gc_x_smooth[:frame_idx+1], gc_y_smooth[:frame_idx+1])
        cart_current.set_offsets([[gc_x_smooth[frame_idx], gc_y_smooth[frame_idx]]])
    
    current_frame = frames_clean[frame_idx] if frame_idx < len(frames_clean) else frames_clean[-1]
    frame_text.set_text(f'Frame: {int(current_frame)}')
    
    return ego_line, cart_line, ego_current, cart_current, frame_text

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=min(len(x_smooth), len(gc_x_smooth)), 
    interval=100, blit=True, repeat=True
)

# Save as GIF
gif_path = "bev_trajectories_animation.gif"
print(f"Saving animation to {gif_path}...")
ani.save(gif_path, writer='pillow', fps=10)
plt.close()

print("Animation complete!")
print(f"Ego trajectory points: {len(x_smooth)}")
print(f"Golf cart trajectory points: {len(gc_x_smooth)}")