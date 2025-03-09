# Face Tracking with Individual Face Windows
import cv2
import mediapipe as mp
import os
import time
import numpy as np
from eye_contact_model import EyeContactDetector  # Import the eye contact detector
import datetime
import threading
import queue

# Init Models
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Class to handle video recording of a face
class FaceVideoRecorder:
    def __init__(self, filename, width, height, fps=20, duration=3.0):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.frame_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None

    def start_recording(self):
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_video)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        print(f"Started recording video to {self.filename}")

    def add_frame(self, frame):
        if self.is_recording:
            # Resize frame if needed
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))
            self.frame_queue.put(frame.copy())

    def _record_video(self):
        # Create video writer with MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))

        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < self.duration:
                try:
                    # Get frame from queue with timeout
                    frame = self.frame_queue.get(timeout=0.1)
                    out.write(frame)
                    frame_count += 1
                except queue.Empty:
                    # If no new frame is available, continue waiting
                    continue
        finally:
            # Release the video writer
            out.release()
            self.is_recording = False
            print(f"Finished recording video: {self.filename} ({frame_count} frames)")

def main():
    # Create directories for eye contact media
    videos_dir = "eye_contact_videos"
    screenshots_dir = "eye_contact_screenshots"
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    # Initialize the eye contact detector
    eye_contact_detector = EyeContactDetector(model_path="models/model_weights.pkl")

    # Dictionary to track eye contact state and last recording time for each face
    eye_contact_tracker = {}

    # Dictionary to track active recorders
    active_recorders = {}

    # Debounce settings
    debounce_time = 5.0  # Seconds between video recordings for the same face
    screenshot_debounce_time = 2.0  # Seconds between screenshots for the same face
    eye_contact_threshold = 0.5  # Threshold for considering eye contact
    video_duration = 4.0  # Duration of recorded videos in seconds
    post_gaze_record_time = 1.0  # Continue recording for this many seconds after eye contact is lost

    # Face box margin settings (percentage of original size)
    face_margin_percent = 40  # Add 40% margin around the face

    print("Starting camera capture...")
    # Use camera index 0 which was confirmed working
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your camera connection.")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit the application")
    print(f"Recording videos and taking screenshots of faces with direct eye contact (threshold: {eye_contact_threshold})")
    print(f"Debounce time between recordings: {debounce_time} seconds")
    print(f"Debounce time between screenshots: {screenshot_debounce_time} seconds")
    print(f"Video duration: {video_duration} seconds")
    print(f"Continue recording after eye contact lost: {post_gaze_record_time} seconds")
    print(f"Face margin: {face_margin_percent}% of original size")
    print(f"Saving videos in MP4 format")

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

                # Calculate margin to add around the face (as a percentage of original size)
                margin_w = int(w * (face_margin_percent / 100))
                margin_h = int(h * (face_margin_percent / 100))

                # Expand the bounding box with the margin
                x_expanded = max(0, x - margin_w)
                y_expanded = max(0, y - margin_h)
                w_expanded = min(w + 2 * margin_w, frame.shape[1] - x_expanded)
                h_expanded = min(h + 2 * margin_h, frame.shape[0] - y_expanded)

                # Extract face region with margin from the original frame
                if w_expanded > 0 and h_expanded > 0:  # Make sure we have a valid region
                    face_img = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded].copy()

                    # Predict eye contact probability
                    eye_contact_prob = eye_contact_detector.predict_eye_contact_probability(face_img)

                    # Determine if there's eye contact based on a threshold
                    has_eye_contact = eye_contact_prob > eye_contact_threshold

                    # Create label with eye contact probability
                    eye_contact_label = f"Eye Contact: {eye_contact_prob:.2f}"

                    # Choose color based on eye contact (green for eye contact, red for no eye contact)
                    eye_contact_color = (0, 255, 0) if has_eye_contact else (0, 0, 255)

                    # Handle video recording and screenshot logic for eye contact
                    current_time = time.time()

                    # Initialize tracker for this face if it doesn't exist
                    if face_id not in eye_contact_tracker:
                        eye_contact_tracker[face_id] = {
                            "last_eye_contact": False,
                            "last_recording_time": 0,
                            "last_screenshot_time": 0,
                            "recording_count": 0,
                            "screenshot_count": 0,
                            "is_recording": False,
                            "lost_eye_contact_time": 0
                        }

                    # Check if we should take a screenshot
                    if (has_eye_contact and
                        (current_time - eye_contact_tracker[face_id]["last_screenshot_time"] > screenshot_debounce_time)):

                        # Create a timestamp for the filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_count = eye_contact_tracker[face_id]["screenshot_count"] + 1
                        screenshot_filename = f"{screenshots_dir}/face_{i+1}_eye_contact_{timestamp}_{screenshot_count}.jpg"

                        # Save the screenshot
                        cv2.imwrite(screenshot_filename, face_img)

                        # Update tracker
                        eye_contact_tracker[face_id]["last_screenshot_time"] = current_time
                        eye_contact_tracker[face_id]["screenshot_count"] = screenshot_count

                        print(f"Screenshot taken for Face {i+1} with eye contact probability {eye_contact_prob:.2f}")

                        # Add a screenshot indicator to the label
                        eye_contact_label = f"Screenshot! {eye_contact_label}"

                    # Check if we should start recording a video
                    if (has_eye_contact and
                        not eye_contact_tracker[face_id]["is_recording"] and
                        (current_time - eye_contact_tracker[face_id]["last_recording_time"] > debounce_time)):

                        # Create a timestamp for the filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        recording_count = eye_contact_tracker[face_id]["recording_count"] + 1
                        video_filename = f"{videos_dir}/face_{i+1}_eye_contact_{timestamp}_{recording_count}.mp4"

                        # Create and start a new recorder
                        recorder = FaceVideoRecorder(
                            filename=video_filename,
                            width=w_expanded,
                            height=h_expanded,
                            fps=20,
                            duration=video_duration
                        )
                        recorder.start_recording()
                        active_recorders[face_id] = recorder

                        # Update tracker
                        eye_contact_tracker[face_id]["last_recording_time"] = current_time
                        eye_contact_tracker[face_id]["recording_count"] = recording_count
                        eye_contact_tracker[face_id]["is_recording"] = True
                        eye_contact_tracker[face_id]["lost_eye_contact_time"] = 0

                        print(f"Started recording video for Face {i+1} with eye contact probability {eye_contact_prob:.2f}")

                        # Add a recording indicator to the label
                        eye_contact_label = f"Recording... {eye_contact_label}"

                    # Track when eye contact is lost during recording
                    if eye_contact_tracker[face_id]["is_recording"]:
                        if has_eye_contact:
                            # Reset the lost eye contact time if eye contact is regained
                            eye_contact_tracker[face_id]["lost_eye_contact_time"] = 0
                        elif eye_contact_tracker[face_id]["lost_eye_contact_time"] == 0:
                            # Start tracking when eye contact was lost
                            eye_contact_tracker[face_id]["lost_eye_contact_time"] = current_time

                    # Add the current frame to the recorder if recording
                    if face_id in active_recorders and active_recorders[face_id].is_recording:
                        active_recorders[face_id].add_frame(face_img)

                    # Check if we should stop recording (only if post-gaze time has elapsed)
                    lost_eye_contact_time = eye_contact_tracker[face_id]["lost_eye_contact_time"]
                    if (eye_contact_tracker[face_id]["is_recording"] and
                        lost_eye_contact_time > 0 and
                        current_time - lost_eye_contact_time > post_gaze_record_time and
                        face_id in active_recorders and
                        not active_recorders[face_id].is_recording):

                        eye_contact_tracker[face_id]["is_recording"] = False
                        # Remove the recorder from active recorders
                        if face_id in active_recorders:
                            del active_recorders[face_id]

                    # Update eye contact state
                    eye_contact_tracker[face_id]["last_eye_contact"] = has_eye_contact

                    # Resize face image to standard size for the window
                    face_img_display = cv2.resize(face_img, (window_width, window_height))

                    # Add eye contact label to the face image
                    cv2.putText(face_img_display, eye_contact_label, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_contact_color, 1)

                    # Add recording indicator if currently recording
                    if eye_contact_tracker[face_id]["is_recording"]:
                        cv2.circle(face_img_display, (window_width - 20, 20), 10, (0, 0, 255), -1)  # Red circle indicator

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
                    cv2.imshow(window_name, face_img_display)

                # Draw rectangle around face on the display frame (not affecting the face windows)
                # Use eye contact color for the rectangle
                rect_color = eye_contact_color if 'eye_contact_color' in locals() else (0, 255, 0)

                # Draw both the original detection rectangle and the expanded rectangle
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 1)  # Original in white
                cv2.rectangle(display_frame, (x_expanded, y_expanded),
                             (x_expanded + w_expanded, y_expanded + h_expanded), rect_color, 2)  # Expanded in color

                # Add face number label on the display frame
                cv2.putText(display_frame, f"Face {i+1}", (x_expanded, y_expanded - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

                # Add eye contact label if available
                if 'eye_contact_label' in locals():
                    cv2.putText(display_frame, eye_contact_label, (x_expanded, y_expanded - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

        # Clean up eye contact tracker and recorders for faces that are no longer detected
        faces_to_remove_from_tracker = []
        for face_id in eye_contact_tracker:
            if face_id not in current_faces:
                faces_to_remove_from_tracker.append(face_id)
                # If there's an active recorder for this face, it will continue until completion

        for face_id in faces_to_remove_from_tracker:
            del eye_contact_tracker[face_id]

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

    # Wait for any active recorders to finish
    for face_id, recorder in active_recorders.items():
        if recorder.recording_thread and recorder.recording_thread.is_alive():
            print(f"Waiting for recording of face {face_id} to complete...")
            recorder.recording_thread.join(timeout=1.0)

    print("Application closed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main application: {e}")