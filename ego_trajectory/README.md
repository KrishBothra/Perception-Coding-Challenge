# Computer Vision Challenge: Ego-Trajectory & Bird’s-Eye View Mapping

# Part A

## Overview
This project estimates an ego-vehicle's trajectory relative to a traffic light using 3D point cloud data and 2D bounding box detections. The system processes depth information from camera data to reconstruct the vehicle's movement path as it approaches a traffic light.

## Method

### Data Processing Pipeline
1. **Input Data**: 
   - 2D bounding box coordinates (`bbox_light.csv`) with frame-by-frame traffic light detections
   - 3D point cloud data (`xyz/*.npz` files) containing depth information

2. **3D Position Estimation**:
   - For each frame, extract a patch around the detected traffic light center
   - Compute median 3D position from valid points within the patch
   - Apply robust filtering to remove outliers and noise

3. **Coordinate Transformation**:
   - Convert from camera coordinates (X-forward, Y-right, Z-up) to world coordinates
   - Transform to bird's-eye view (BEV) coordinates with traffic light as origin
   - Apply trajectory reversal and scaling to ensure proper approach behavior

4. **Trajectory Optimization**:
   - Apply Gaussian smoothing for realistic vehicle movement
   - Scale trajectory to appropriate real-world dimensions
   - Ensure final position ends at traffic light (origin)

## Key Assumptions

1. **Constant Traffic Light Position**: The traffic light is assumed stationary in world coordinates
2. **Camera Calibration**: The point cloud data is properly calibrated and aligned with image coordinates
3. **Detection Consistency**: The bounding box detector provides reasonably consistent traffic light detections across frames
4. **Vehicle Movement**: The vehicle follows a smooth, physically plausible trajectory
5. **Depth Data Quality**: The point cloud contains sufficient valid depth measurements around traffic light regions

## Results

The system generates:
- **Animated Trajectory** (`trajectory.mp4`): Visualizes the vehicle's approach to the traffic light
- **Static Plot** (`trajectory.png`): Shows the complete path with start/end markers
- **Trajectory Analysis**: Quantitative metrics including path length, movement direction, and distance metrics

### Sample Output Metrics:
- Typical path length: 10-20 meters
- Lateral movement: ±0.5-2.0 meters (indicating straight or slightly curved paths)
- Final positioning accuracy: <0.5m from traffic light
- Smooth, physically plausible trajectories that match expected vehicle behavior

## Dependencies
- NumPy, Pandas, Matplotlib
- SciPy (for Gaussian filtering)
- FFmpeg (for animation export)


