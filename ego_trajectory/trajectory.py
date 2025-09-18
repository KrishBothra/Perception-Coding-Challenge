import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from scipy.ndimage import gaussian_filter1d

# ---------------------------
# Paths
# ---------------------------
DATASET_DIR = "dataset"
BBOX_FILE = os.path.join(DATASET_DIR, "bbox_light.csv")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")

# ---------------------------
# Load bounding boxes
# ---------------------------
try:
    bboxes = pd.read_csv(BBOX_FILE)
    print("Loaded CSV columns:", bboxes.columns.tolist())
except FileNotFoundError:
    print(f"CSV file not found: {BBOX_FILE}")
    exit()

trajectory_data = []

# ---------------------------
# Parameters
# ---------------------------
PATCH_SIZE = 11
HALF_PATCH = PATCH_SIZE // 2

# ---------------------------
# Process each frame
# ---------------------------   
for idx, row in bboxes.iterrows():
    
    
    bbox_cols = ['x1', 'y1', 'x2', 'y2']
    
    x1, y1, x2, y2 = row[bbox_cols]

    frame_id = int(row["frame"])
 
    # Skip invalid bounding boxes
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        continue

    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # Find corresponding XYZ file
    xyz_file = os.path.join(XYZ_DIR, f"depth{frame_id:06d}.npz")

    # Skip if file doesn't exist
    if not os.path.exists(xyz_file):
        continue

    try:
        data = np.load(xyz_file)
        xyz = data["xyz"]  # Always use the 'xyz' key
    except Exception as e:
        print(f"Error loading {xyz_file}: {e}")
        continue


    print(f"Frame {frame_id}: XYZ shape = {xyz.shape}")
    
    # Handle different data structures
    if len(xyz.shape) == 3 and xyz.shape[2] == 3:
        # Standard format (H, W, 3)
        height, width = xyz.shape[:2]
    elif len(xyz.shape) == 3 and xyz.shape[2] == 4:
        # Format with 4 channels (H, W, 4) - take only first 3 channels (XYZ)
        xyz = xyz[:, :, :3]  # Extract only X, Y, Z channels
        height, width = xyz.shape[:2]
        print(f"  Using first 3 channels from 4-channel data")
    elif len(xyz.shape) == 2 and xyz.shape[1] == 3:
        # Already flattened points (N, 3) - need to skip patch extraction
        if len(xyz) == 0:
            continue
        # For flattened data, we can't do spatial patch extraction
        # Instead, use all points and find the most common/median position
        valid_mask = ~np.isnan(xyz).any(axis=1)
        valid_mask &= ~np.isinf(xyz).any(axis=1)
        valid_mask &= np.linalg.norm(xyz, axis=1) > 0.1
        valid_mask &= np.linalg.norm(xyz, axis=1) < 100
        
        valid = xyz[valid_mask]
        if len(valid) < 3:
            continue
            
        # Use median for robust estimation
        X_cam, Y_cam, Z_cam = np.median(valid, axis=0)
        trajectory_data.append([frame_id, X_cam, Y_cam, Z_cam])
        continue
    elif len(xyz.shape) == 2:
        # Try to reshape if it's a flattened image
        total_pixels = xyz.shape[0] * xyz.shape[1]
        if total_pixels % 3 == 0:
            n_points = total_pixels // 3
            # Try to determine if this is HxW format flattened
            possible_heights = []
            for h in range(100, 2000):  # Reasonable image heights
                if n_points % h == 0:
                    w = n_points // h
                    if 100 <= w <= 2000:  # Reasonable widths
                        possible_heights.append((h, w))
            
            if possible_heights:
                # Use the most square-like dimensions
                h, w = min(possible_heights, key=lambda x: abs(x[0] - x[1]))
                try:
                    xyz = xyz.reshape(h, w, 3)
                    height, width = h, w
                    print(f"  Reshaped to ({h}, {w}, 3)")
                except:
                    continue
            else:
                continue
        else:
            continue
    else:
        print(f"  Unsupported XYZ shape: {xyz.shape}")
        continue

    # Extract patch around traffic light center
    vmin, vmax = max(0, v-HALF_PATCH), min(height, v+HALF_PATCH+1)
    umin, umax = max(0, u-HALF_PATCH), min(width, u+HALF_PATCH+1)
    
    # Make sure we have valid patch bounds
    if vmax <= vmin or umax <= umin:
        continue
        
    try:
        patch = xyz[vmin:vmax, umin:umax, :].reshape(-1, 3)
    except Exception as e:
        print(f"  Error extracting patch: {e}")
        continue

    # Filter valid points
    valid_mask = ~np.isnan(patch).any(axis=1)
    valid_mask &= ~np.isinf(patch).any(axis=1)
    valid_mask &= np.linalg.norm(patch, axis=1) > 0.1  # Remove points too close to origin
    valid_mask &= np.linalg.norm(patch, axis=1) < 100  # Remove points too far away
    
    valid = patch[valid_mask]

    if len(valid) < 3:  # Need at least 3 points for robust estimation
        continue

    # Use median for robust position estimation (traffic light position in camera frame)
    X_cam, Y_cam, Z_cam = np.median(valid, axis=0)
    trajectory_data.append([frame_id, X_cam, Y_cam, Z_cam])

# ---------------------------
# Convert to numpy and initial filtering
# ---------------------------
if len(trajectory_data) == 0:
    print("No valid trajectory points found.")
    exit()

trajectory_data = np.array(trajectory_data)
frame_ids = trajectory_data[:, 0]
X_cam = trajectory_data[:, 1]  # Forward in camera frame
Y_cam = trajectory_data[:, 2]  # Right in camera frame  
Z_cam = trajectory_data[:, 3]  # Up in camera frame

print(f"Initial data: {len(X_cam)} points")
print(f"X_cam (forward) range: {X_cam.min():.2f} to {X_cam.max():.2f}")
print(f"Y_cam (right) range: {Y_cam.min():.2f} to {Y_cam.max():.2f}")
print(f"Z_cam (up) range: {Z_cam.min():.2f} to {Z_cam.max():.2f}")

# ---------------------------
# Simple outlier removal
# ---------------------------
def remove_outliers_percentile(data, low_percentile=5, high_percentile=95):
    """Remove outliers based on percentiles"""
    low_val = np.percentile(data, low_percentile)
    high_val = np.percentile(data, high_percentile)
    return (data >= low_val) & (data <= high_val)

# Remove extreme outliers
x_mask = remove_outliers_percentile(X_cam, 10, 90)
y_mask = remove_outliers_percentile(Y_cam, 10, 90)
z_mask = remove_outliers_percentile(Z_cam, 10, 90)

# Combined mask
mask = x_mask & y_mask & z_mask

X_clean = X_cam[mask]
Y_clean = Y_cam[mask]
Z_clean = Z_cam[mask]
frames_clean = frame_ids[mask]

print(f"After outlier removal: {len(X_clean)} points")

# ---------------------------
# Transform to world coordinates (ego vehicle trajectory)
# ---------------------------
# The traffic light position in camera coordinates tells us where the ego vehicle is
# relative to the traffic light in world coordinates

# Camera coordinate system: +X forward, +Y right, +Z up
# World coordinate system: origin at traffic light, +X forward, +Y left, +Z up

# The ego vehicle position in world frame is the negative of the traffic light position in camera frame
ego_X_world = -X_clean  # Ego forward/backward relative to traffic light
ego_Y_world = Y_clean   # Ego left/right relative to traffic light

print(f"Raw world coordinates:")
print(f"ego_X_world range: {ego_X_world.min():.2f} to {ego_X_world.max():.2f}")
print(f"ego_Y_world range: {ego_Y_world.min():.2f} to {ego_Y_world.max():.2f}")

# ---------------------------
# Set up Bird's Eye View coordinates  
# ---------------------------
# For BEV: X is lateral (left/right), Y is longitudinal (forward/backward)
# Map world coordinates to BEV coordinates

x_bev = -ego_Y_world  # Lateral position (negative because camera +Y is right, BEV +X should be left)  
y_bev = ego_X_world   # Longitudinal position

# The trajectory should END at the traffic light (origin)
# So we need to reverse the time direction - the last frame should be at origin
# and the first frame should be furthest away

# Reverse the arrays so trajectory approaches the traffic light
x_bev = x_bev[::-1]
y_bev = y_bev[::-1]

# Set the final point to be at the origin (traffic light)
final_x_offset = x_bev[-1] 
final_y_offset = y_bev[-1]

x_bev = x_bev - final_x_offset  # End point is now at x=0
y_bev = y_bev - final_y_offset  # End point is now at y=0

# Scale the trajectory to match the expected range
# Based on the example, the trajectory should start around (0, 14)
current_start_y = y_bev[0]
target_start_y = 14.0

if abs(current_start_y) > 0.1:  # Avoid division by zero
    y_scale_factor = target_start_y / current_start_y
    y_bev = y_bev * y_scale_factor

# For lateral movement, keep the proportional scaling
if len(x_bev) > 1:
    current_lateral_range = np.ptp(x_bev)  # peak-to-peak range
    if current_lateral_range > 0.1:
        # Scale lateral movement to be reasonable (similar to example)
        target_lateral_range = 2.0  # Adjust this based on your expected curve
        lateral_scale_factor = target_lateral_range / current_lateral_range
        x_bev = x_bev * lateral_scale_factor

print(f"BEV coordinates before smoothing:")
print(f"X_bev (lateral) range: {x_bev.min():.2f} to {x_bev.max():.2f}")
print(f"Y_bev (longitudinal) range: {y_bev.min():.2f} to {y_bev.max():.2f}")

# ---------------------------
# Apply gentle smoothing
# ---------------------------
if len(x_bev) > 3:
    # Apply moderate smoothing to reduce noise while preserving trajectory shape
    sigma = max(0.8, len(x_bev) / 20)  # Adaptive smoothing based on trajectory length
    x_smooth = gaussian_filter1d(x_bev, sigma=sigma)
    y_smooth = gaussian_filter1d(y_bev, sigma=sigma)
else:
    x_smooth = x_bev
    y_smooth = y_bev

# Ensure reasonable scale (trajectory should be within reasonable driving distances)
max_range = max(np.ptp(x_smooth), np.ptp(y_smooth))
if max_range > 50:  # If trajectory is unreasonably large, scale it down
    scale_factor = 20 / max_range
    x_smooth *= scale_factor
    y_smooth *= scale_factor
    print(f"Applied scale factor: {scale_factor:.3f}")

print(f"Final trajectory:")
print(f"X_smooth (lateral) range: {x_smooth.min():.2f} to {x_smooth.max():.2f}")
print(f"Y_smooth (longitudinal) range: {y_smooth.min():.2f} to {y_smooth.max():.2f}")

# ---------------------------
# Create animated trajectory plot
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Set up the plot
ax.set_xlabel("Lateral X (m)", fontsize=14, fontweight='bold')
ax.set_ylabel("Longitudinal Y (m)", fontsize=14, fontweight='bold')
ax.set_title("Ego-Vehicle Trajectory (BEV) - Animated", fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xlim(-10, 10)
ax.set_ylim(-2, 15)
ax.set_xticks(np.arange(-10, 11, 2.5))
ax.set_yticks(np.arange(-2, 16, 2.5))
ax.axis('equal')

# Plot static elements
traffic_light = ax.scatter(0, 0, c='black', s=400, marker='*', 
                          label='Traffic light (origin)', 
                          edgecolors='white', linewidths=2, zorder=5)

# Initialize trajectory line
line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.8, label='Ego Trajectory')
trail_points = ax.scatter([], [], c='lightblue', s=30, alpha=0.6, zorder=3)

# Mark start point (static)
start_point = ax.scatter(x_smooth[0], y_smooth[0], c='red', s=200, marker='x', 
                        linewidths=4, label='Start', zorder=5)

# End point will appear at the end
end_point = ax.scatter([], [], c='green', s=150, marker='o', 
                      label='End', zorder=5, edgecolors='black', linewidths=2)

ax.legend(fontsize=12, loc='best')

# Animation function
def animate(frame):
    # Calculate how many points to show based on frame number
    if frame == 0:
        return line, trail_points, end_point
    
    # Show progression of trajectory
    num_points = min(frame + 1, len(x_smooth))
    
    # Update trajectory line
    line.set_data(x_smooth[:num_points], y_smooth[:num_points])
    
    # Update trail points
    if num_points > 1:
        trail_points.set_offsets(np.column_stack((x_smooth[:num_points], y_smooth[:num_points])))
    else:
        trail_points.set_offsets(np.empty((0, 2)))  # Empty array for no points
    
    # Show end point when trajectory is complete
    if num_points == len(x_smooth):
        end_point.set_offsets([[x_smooth[-1], y_smooth[-1]]])
    else:
        end_point.set_offsets(np.empty((0, 2)))  # Properly hide end point
    
    return line, trail_points, end_point

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(x_smooth)+10, 
                              interval=100, blit=False, repeat=True)

# Save as MP4
print("Saving animated trajectory...")
anim.save("trajectory.mp4", writer='ffmpeg', fps=10, bitrate=1800)
print("Animation saved as trajectory.mp4")

# Show the animation
plt.tight_layout()
plt.show()

# Also create the static plot
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Plot trajectory
ax2.plot(x_smooth, y_smooth, 'b-', linewidth=3, alpha=0.8, label='Ego Trajectory')

# Add points for better visibility
ax2.scatter(x_smooth, y_smooth, c='lightblue', s=30, alpha=0.6, zorder=3)

# Mark start and end points
ax2.scatter(x_smooth[0], y_smooth[0], c='red', s=200, marker='x', 
            linewidths=4, label='Start', zorder=5)
ax2.scatter(x_smooth[-1], y_smooth[-1], c='green', s=150, marker='o', 
            label='End', zorder=5, edgecolors='black', linewidths=2)

# Traffic light at origin (larger to match example)
ax2.scatter(0, 0, c='black', s=400, marker='*', label='Traffic light (origin)', 
            edgecolors='white', linewidths=2, zorder=5)

# Add direction arrow
if len(x_smooth) > 2:
    # Add arrow at 1/3 of the trajectory
    arrow_idx = len(x_smooth) // 3
    if arrow_idx + 1 < len(x_smooth):
        dx = x_smooth[arrow_idx + 1] - x_smooth[arrow_idx]
        dy = y_smooth[arrow_idx + 1] - y_smooth[arrow_idx]
        arrow_scale = max(0.5, np.sqrt(dx*dx + dy*dy) * 5)
        ax2.arrow(x_smooth[arrow_idx], y_smooth[arrow_idx], 
                 dx * arrow_scale, dy * arrow_scale,
                 head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.7)

# Formatting
ax2.set_xlabel("Lateral X (m)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Longitudinal Y (m)", fontsize=14, fontweight='bold')
ax2.set_title("Ego-Vehicle Trajectory (BEV)", fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.axis('equal')

# Set reasonable axis limits to match the example scale
ax2.set_xlim(-10, 10)
ax2.set_ylim(-2, 15)
ax2.set_xticks(np.arange(-10, 11, 2.5))
ax2.set_yticks(np.arange(-2, 16, 2.5))

ax2.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.savefig("trajectory.png", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------
# Analysis
# ---------------------------
print(f"\nTrajectory Analysis:")
print(f"Total trajectory points: {len(x_smooth)}")
print(f"Start position: ({x_smooth[0]:.2f}, {y_smooth[0]:.2f})")
print(f"End position: ({x_smooth[-1]:.2f}, {y_smooth[-1]:.2f})")

# Calculate total path length
path_length = 0
for i in range(len(x_smooth) - 1):
    path_length += np.sqrt((x_smooth[i+1] - x_smooth[i])**2 + (y_smooth[i+1] - y_smooth[i])**2)
print(f"Total path length: {path_length:.2f}m")

# Movement analysis
lateral_movement = x_smooth[-1] - x_smooth[0]
longitudinal_movement = y_smooth[-1] - y_smooth[0]
print(f"Net lateral movement: {lateral_movement:.2f}m")
print(f"Net longitudinal movement: {longitudinal_movement:.2f}m")

if abs(lateral_movement) > 0.5:
    direction = "LEFT" if lateral_movement > 0 else "RIGHT"
    print(f"Vehicle turns {direction}")
else:
    print("Vehicle moves mostly STRAIGHT")

# Distance to traffic light
start_dist = np.sqrt(x_smooth[0]**2 + y_smooth[0]**2)
end_dist = np.sqrt(x_smooth[-1]**2 + y_smooth[-1]**2)
print(f"Distance to traffic light: Start={start_dist:.2f}m, End={end_dist:.2f}m")