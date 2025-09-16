import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
DATASET_DIR = "dataset"
BBOX_FILE = os.path.join(DATASET_DIR, "bbox_light.csv")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")

# Load bounding boxes
bboxes = pd.read_csv(BBOX_FILE)

trajectory = []

for idx, row in bboxes.iterrows():
    frame_id = int(row["frame"])

    # Skip invalid bounding boxes (all zeros)
    if row["x1"] == 0 and row["y1"] == 0 and row["x2"] == 0 and row["y2"] == 0:
        print(f"Frame {frame_id}: no bbox, skipping")
        continue

    # Bounding box center (pixel coords)
    u = int((row["x1"] + row["x2"]) // 2)
    v = int((row["y1"] + row["y2"]) // 2)

    # --- FIXED file name pattern ---
    xyz_file = os.path.join(XYZ_DIR, f"depth{frame_id:06d}.npz")
    if not os.path.exists(xyz_file):
        print(f"Missing file: {xyz_file}")
        continue

    xyz = np.load(xyz_file)["xyz"]

    # --- Use a patch around the pixel ---
    patch_size = 9
    half = patch_size // 2

    vmin, vmax = max(0, v-half), min(xyz.shape[0], v+half+1)
    umin, umax = max(0, u-half), min(xyz.shape[1], u+half+1)

    patch = xyz[vmin:vmax, umin:umax, :].reshape(-1, 3)

    # Filter out invalid points
    valid = patch[~np.isnan(patch).any(axis=1)]
    valid = valid[np.linalg.norm(valid, axis=1) > 1e-6]

    if len(valid) == 0:
        print(f"Frame {frame_id}: no valid depth in patch")
        continue

    # Median is more robust than mean
    X, Y, Z = np.median(valid, axis=0)
    trajectory.append((X, Y, Z))

    print(f"Frame {frame_id}: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

# Convert to numpy



trajectory = np.array(trajectory)  # shape (N, 3): X, Y, Z

if trajectory.shape[0] == 0:
    print("No valid trajectory points found.")
else:
    lateral = trajectory[:, 0]   # X (side-to-side)
    forward = trajectory[:, 2]   # Z (forward)

    # Remove outliers (optional)
    diffs = np.sqrt(np.diff(lateral, prepend=lateral[0])**2 +
                    np.diff(forward, prepend=forward[0])**2)
    mask = diffs < 1.0
    lateral = lateral[mask]
    forward = forward[mask]

    plt.figure(figsize=(8, 8))
    plt.plot(lateral, forward, "b-", lw=2, label="Ego trajectory")
    plt.scatter(lateral[0], forward[0], c="red", s=100, marker="x", label="Start")
    plt.scatter(lateral[-1], forward[-1], c="green", s=100, marker="o", label="End")
    plt.scatter(0, 0, c="black", s=150, marker="*", label="Origin")

    # Labels
    plt.xlabel("Lateral (X, m)")
    plt.ylabel("Forward (Y, m)")
    plt.title("Ego-Vehicle Trajectory (World Frame)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend(loc="upper right")

    # Axes formatting
    xmax = np.max(np.abs(lateral))
    plt.xlim(-xmax, xmax)             # symmetric about 0
    plt.ylim(0, np.max(forward))      # start forward axis at 0

    plt.show()

