# Head Pose Estimation Using Mediapipe and OpenCV

This project implements real-time **head pose estimation** using **Mediapipe**'s Face Mesh and **OpenCV**. The program detects facial landmarks, estimates the orientation of the head in 3D space, and visualizes the head pose using axes overlaid on a live webcam feed.

---

## Features

- Real-time face detection and landmark tracking using Mediapipe Face Mesh.
- Head pose estimation via 3D-2D point correspondence and `cv2.solvePnP`.
- Visualizes head orientation with overlaid X, Y, and Z axes.

---

## Prerequisites

1. Python 3.7+
2. Libraries:
   - OpenCV
   - Mediapipe
   - Numpy

Install dependencies using:

```bash
pip install opencv-python mediapipe numpy
```

---

## Usage

1. Clone the repository or save the code to a file (e.g., `head_pose_estimation.py`).
2. Run the script:

   ```bash
   python head_pose_estimation.py
   ```

3. Grant webcam access when prompted.
4. Observe the live head pose estimation with axes drawn on your face.
5. Press `q` to exit the application.

---

## How It Works

1. **Face Detection**:
   - Mediapipe's Face Mesh detects 468 facial landmarks in real-time.
2. **3D Model Points**:
   - Predefined 3D model points represent key facial landmarks (e.g., nose tip, chin, eyes).
3. **Head Pose Estimation**:
   - OpenCV's `cv2.solvePnP` computes the head's rotation and translation vectors.
4. **Visualization**:
   - The rotation is visualized by projecting 3D axes onto the image plane and overlaying them on the face.

---

## 3D Model Points Mapping

The following Mediapipe Face Mesh landmarks are used for head pose estimation:

| Landmark             | Index |
|-----------------------|-------|
| Nose tip             | 1     |
| Chin                | 199   |
| Left eye corner     | 33    |
| Right eye corner    | 263   |
| Left mouth corner   | 61    |
| Right mouth corner  | 291   |

---

## Example Output

When the script is running, you will see:

- **Red Line (X-axis)**: Indicates left-right tilt.
- **Green Line (Y-axis)**: Indicates up-down tilt.
- **Blue Line (Z-axis)**: Points outward, indicating the facing direction.

---

## Customization

- **Change Camera Resolution**:
  Modify the `camera_matrix` to match your webcam's resolution.
- **Use Different Landmarks**:
  Adjust the `model_points` and `get_image_points` functions for additional landmarks.
- **Add Features**:
  Extend the project with gesture recognition or attention tracking.

---

## Troubleshooting

- **Webcam Not Detected**:
  Ensure the webcam is properly connected and accessible.
- **Low Accuracy**:
  Check the `camera_matrix` and `model_points` values for accuracy.
- **Dependencies Not Found**:
  Ensure all required libraries are installed.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more information.


---

Enjoy experimenting with head pose estimation! 🎉

