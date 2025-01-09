import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Define 3D model points corresponding to Mediapipe landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip (Landmark 1)
    (0.0, -330.0, -65.0),   # Chin (Landmark 199)
    (-225.0, 170.0, -135.0),# Left eye left corner (Landmark 33)
    (225.0, 170.0, -135.0), # Right eye right corner (Landmark 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (Landmark 61)
    (150.0, -150.0, -125.0)  # Right mouth corner (Landmark 291)
], dtype="double")

# Camera intrinsic parameters (assuming 640x480 resolution)
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# Distortion coefficients (assuming no distortion)
dist_coeffs = np.zeros((4, 1))

def get_image_points(landmarks, image_height, image_width):
    """Extract relevant 2D points from Mediapipe landmarks."""
    image_points = np.array([
        (landmarks[1].x * image_width, landmarks[1].y * image_height),     # Nose tip
        (landmarks[199].x * image_width, landmarks[199].y * image_height), # Chin
        (landmarks[33].x * image_width, landmarks[33].y * image_height),   # Left eye left corner
        (landmarks[263].x * image_width, landmarks[263].y * image_height), # Right eye right corner
        (landmarks[61].x * image_width, landmarks[61].y * image_height),   # Left mouth corner
        (landmarks[291].x * image_width, landmarks[291].y * image_height)  # Right mouth corner
    ], dtype="double")
    return image_points

# Initialize Mediapipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            image_points = get_image_points(landmarks, h, w)

            # SolvePnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)

            # Project 3D points to 2D to draw axes
            axis_points = np.float32([
                [0, 0, 500],   # Nose direction (Z-axis)
                [500, 0, 0],   # X-axis
                [0, 500, 0]    # Y-axis
            ])
            axis_2d_points, _ = cv2.projectPoints(
                axis_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # Convert points to integer tuples
            nose_tip = tuple(image_points[0].astype(int))
            x_axis = tuple(axis_2d_points[1].ravel().astype(int))
            y_axis = tuple(axis_2d_points[2].ravel().astype(int))
            z_axis = tuple(axis_2d_points[0].ravel().astype(int))

            # Draw the axes on the frame
            cv2.line(frame, nose_tip, x_axis, (0, 0, 255), 2)  # X-axis in red
            cv2.line(frame, nose_tip, y_axis, (0, 255, 0), 2)  # Y-axis in green
            cv2.line(frame, nose_tip, z_axis, (255, 0, 0), 2)  # Z-axis in blue

    # Display the frame
    cv2.imshow("Head Pose Estimation with Mediapipe", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
