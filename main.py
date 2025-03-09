# Face Tracking with Individual Face Windows
import cv2
import mediapipe as mp
import os
import time
import numpy as np
from datetime import datetime

# Init Models
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def calculate_gaze_direction(face_landmarks, image_shape):
    """
    Calculate gaze direction using multiple eye landmarks and pupil position
    Returns a score where lower values indicate direct gaze
    """
    # Get key landmarks for gaze detection
    # Eye landmarks
    left_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
    right_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

    # Calculate eye centers
    left_eye_center = {
        'x': sum(landmark.x for landmark in left_eye_landmarks) / len(left_eye_landmarks),
        'y': sum(landmark.y for landmark in left_eye_landmarks) / len(left_eye_landmarks)
    }

    right_eye_center = {
        'x': sum(landmark.x for landmark in right_eye_landmarks) / len(right_eye_landmarks),
        'y': sum(landmark.y for landmark in right_eye_landmarks) / len(right_eye_landmarks)
    }

    # Get pixel coordinates for visualization
    h, w = image_shape[:2]
    left_eye_px = (int(left_eye_center['x'] * w), int(left_eye_center['y'] * h))
    right_eye_px = (int(right_eye_center['x'] * w), int(right_eye_center['y'] * h))

    # Basic gaze detection using eye symmetry and face orientation
    # When looking at camera, eyes should be symmetrical and face should be frontal

    # 1. Eye symmetry - horizontal alignment
    eye_y_diff = abs(left_eye_center['y'] - right_eye_center['y'])
    eye_x_diff = abs(left_eye_center['x'] - right_eye_center['x'])

    # 2. Use face landmarks for orientation
    # Nose tip (1) should be centered between eyes for direct gaze
    try:
        nose_tip = face_landmarks.landmark[1]
        nose_x = nose_tip.x

        # Calculate horizontal center between eyes
        eye_center_x = (left_eye_center['x'] + right_eye_center['x']) / 2

        # Nose should be centered between eyes when looking at camera
        nose_offset = abs(nose_x - eye_center_x)

        # 3. Use eyebrow positions for vertical gaze
        left_eyebrow = face_landmarks.landmark[107].y
        right_eyebrow = face_landmarks.landmark[336].y
        eyebrow_eye_ratio = ((left_eyebrow - left_eye_center['y']) +
                             (right_eyebrow - right_eye_center['y'])) / 2

        # Combine metrics - lower is more direct
        gaze_score = (eye_y_diff * 2) + (nose_offset * 3) + (eyebrow_eye_ratio * 0.5)

        # Print debug info
        print(f"Eye Y diff: {eye_y_diff:.4f}, Nose offset: {nose_offset:.4f}, Eyebrow ratio: {eyebrow_eye_ratio:.4f}")
        print(f"Gaze score: {gaze_score:.4f}")

    except Exception as e:
        print(f"Error in gaze calculation: {e}")
        # Fallback to basic eye position
        gaze_score = eye_y_diff + eye_x_diff

    return gaze_score, left_eye_px, right_eye_px

def main():
    # Create faces directory if it doesn't exist
    os.makedirs("faces", exist_ok=True)

    print("Starting camera capture...")
    # Use camera index 0 which was confirmed working
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your camera connection.")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit the application")

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize MediaPipe face detection
    # Use model_selection=1 for full range detection (better for multiple faces)
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    # Initialize face mesh for gaze detection
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,  # Detect up to 5 faces
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    start_time = time.time()

    # Dictionary to track active face windows
    active_faces = {}

    # Dictionary to track gaze scores
    gaze_scores = {}

    # Window positioning parameters
    main_window_name = 'Face Detection'
    window_width = 200  # Width of face windows
    window_height = 200  # Height of face windows

    # Position the main window
    cv2.namedWindow(main_window_name)
    cv2.moveWindow(main_window_name, 50, 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame - camera disconnected or end of video file")
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        if frame_count % 30 == 0:  # Print every 30 frames
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}, Frame size: {frame.shape}")

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect Faces
        faces = face_detection.process(rgb_frame)

        # Process with face mesh (for eye landmarks)
        face_mesh_results = face_mesh.process(rgb_frame)

        # Track which faces we've seen in this frame
        current_faces = set()

        # Create a copy of the frame for drawing indicators
        display_frame = frame.copy()

        # Store face bounding boxes
        face_boxes = []

        # Draw face detections and create face windows
        if faces.detections:
            num_faces = len(faces.detections)
            print(f"Detected {num_faces} faces")

            for i, detection in enumerate(faces.detections):
                # Get face ID (using index for now)
                face_id = f"face_{i+1}"
                current_faces.add(face_id)

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0])
                w, h = int(bbox.width * frame.shape[1]), int(bbox.height * frame.shape[0])

                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                # Store face box
                face_boxes.append((x, y, w, h))

                # Extract face region from the original frame (before drawing rectangles)
                if w > 0 and h > 0:  # Make sure we have a valid region
                    face_img = frame[y:y+h, x:x+w].copy()

                    # Resize face image to standard size for the window
                    face_img = cv2.resize(face_img, (window_width, window_height))

                    # Add gaze score to the face image if available
                    if face_id in gaze_scores:
                        gaze_score = gaze_scores[face_id]

                        # Calculate gaze probability - new formula
                        # Lower score is better (more direct gaze)
                        # Scale to make it more sensitive in the 0-0.2 range
                        # Clamp between 0-100%
                        gaze_probability = max(0, min(100, 100 * (1.0 - min(1.0, gaze_score * 5.0))))

                        # Add text showing gaze probability
                        cv2.putText(face_img, f"Direct Gaze: {gaze_probability:.1f}%",
                                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # Add visual indicator (red if staring, green if not)
                        if gaze_probability > 70:  # High probability of direct gaze
                            cv2.putText(face_img, "DIRECT GAZE", (10, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Create or update face window
                    window_name = f"Face {i+1}"

                    # Position the window (main window + offset based on face number)
                    window_x = 50 + display_frame.shape[1] + 20  # Main window width + margin
                    window_y = 50 + (i * (window_height + 30))  # Vertical offset for each face

                    # Create named window and position it
                    if face_id not in active_faces:
                        cv2.namedWindow(window_name)
                        cv2.moveWindow(window_name, window_x, window_y)
                        active_faces[face_id] = window_name

                    # Display face in its window
                    cv2.imshow(window_name, face_img)

                # Draw rectangle around face on the display frame (not affecting the face windows)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Add face number label on the display frame
                cv2.putText(display_frame, f"Face {i+1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Process eye landmarks if face mesh is detected
        if face_mesh_results.multi_face_landmarks:
            num_meshes = len(face_mesh_results.multi_face_landmarks)
            print(f"Detected {num_meshes} face meshes")

            for i, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                try:
                    # Calculate gaze using the improved function
                    gaze_score, left_eye_px, right_eye_px = calculate_gaze_direction(face_landmarks, display_frame.shape)

                    # Store gaze score for this face
                    face_id = f"face_{i+1}"
                    gaze_scores[face_id] = gaze_score

                    # Draw eye centers on the display frame (not on the face windows)
                    cv2.circle(display_frame, left_eye_px, 3, (255, 0, 0), -1)
                    cv2.circle(display_frame, right_eye_px, 3, (255, 0, 0), -1)

                    # Calculate gaze probability for display
                    gaze_probability = max(0, min(100, 100 * (1.0 - min(1.0, gaze_score * 5.0))))

                    # Add gaze score to main display for debugging
                    cv2.putText(display_frame, f"Gaze {i+1}: {gaze_probability:.1f}%",
                               (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Also show raw score for debugging
                    cv2.putText(display_frame, f"Score: {gaze_score:.4f}",
                               (10, 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                except Exception as e:
                    print(f"Error processing landmarks for face {i+1}: {e}")

        # Close windows for faces that are no longer detected
        faces_to_remove = []
        for face_id in active_faces:
            if face_id not in current_faces:
                window_name = active_faces[face_id]
                cv2.destroyWindow(window_name)
                faces_to_remove.append(face_id)

        # Remove closed windows from tracking
        for face_id in faces_to_remove:
            del active_faces[face_id]
            if face_id in gaze_scores:
                del gaze_scores[face_id]

        # Add instructions text
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the main frame
        cv2.imshow(main_window_name, display_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application...")
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main application: {e}")