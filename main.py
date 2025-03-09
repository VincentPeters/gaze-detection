# Face Tracking with Individual Face Windows
import cv2
import mediapipe as mp
import os
import time
import numpy as np
from eye_contact_model import EyeContactDetector  # Import the eye contact detector

# Init Models
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():
    # Create faces directory if it doesn't exist
    os.makedirs("faces", exist_ok=True)

    # Initialize the eye contact detector
    eye_contact_detector = EyeContactDetector(model_path="models/model_weights.pkl")

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

    frame_count = 0
    start_time = time.time()

    # Dictionary to track active face windows
    active_faces = {}

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

        # Track which faces we've seen in this frame
        current_faces = set()

        # Create a copy of the frame for drawing indicators
        display_frame = frame.copy()

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

                # Extract face region from the original frame (before drawing rectangles)
                if w > 0 and h > 0:  # Make sure we have a valid region
                    face_img = frame[y:y+h, x:x+w].copy()

                    # Predict eye contact probability
                    eye_contact_prob = eye_contact_detector.predict_eye_contact_probability(face_img)

                    # Determine if there's eye contact based on a threshold
                    has_eye_contact = eye_contact_prob > 0.5

                    # Create label with eye contact probability
                    eye_contact_label = f"Eye Contact: {eye_contact_prob:.2f}"

                    # Choose color based on eye contact (green for eye contact, red for no eye contact)
                    eye_contact_color = (0, 255, 0) if has_eye_contact else (0, 0, 255)

                    # Resize face image to standard size for the window
                    face_img = cv2.resize(face_img, (window_width, window_height))

                    # Add eye contact label to the face image
                    cv2.putText(face_img, eye_contact_label, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_contact_color, 1)

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
                # Use eye contact color for the rectangle
                rect_color = eye_contact_color if 'eye_contact_color' in locals() else (0, 255, 0)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), rect_color, 2)

                # Add face number label on the display frame
                cv2.putText(display_frame, f"Face {i+1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

                # Add eye contact label if available
                if 'eye_contact_label' in locals():
                    cv2.putText(display_frame, eye_contact_label, (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

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