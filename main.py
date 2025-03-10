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
import config  # Import the configuration file
from config_window import ConfigWindow  # Import the configuration window

# Init Models
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Class to handle video recording of a face
class FaceVideoRecorder:
    def __init__(self, filename, width, height, fps=config.VIDEO_FPS, duration=config.VIDEO_DURATION):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.frame_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.last_frame = None  # Store the last frame for when detection fails

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
            self.last_frame = frame.copy()  # Store this frame

    def add_last_frame_if_available(self):
        """Add the last frame again if available (used when face detection fails)"""
        if self.is_recording and self.last_frame is not None:
            self.frame_queue.put(self.last_frame.copy())

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
    # Create directories if they don't exist
    os.makedirs(config.FACES_DIR, exist_ok=True)
    os.makedirs(config.VIDEOS_DIR, exist_ok=True)
    os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

    # Initialize the eye contact detector
    eye_contact_detector = EyeContactDetector(model_path=config.MODEL_PATH)

    # Dictionary to track eye contact state and last recording time for each face
    eye_contact_tracker = {}

    # Dictionary to track active recorders
    active_recorders = {}

    # Dictionary to track face positions when detection fails
    last_face_positions = {}

    # Initialize configuration window
    config_window = ConfigWindow()

    print("Starting camera capture...")
    # Use camera index 0 which was confirmed working
    cap = cv2.VideoCapture(0)

    # Try to set the highest possible resolution for the camera
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    highest_width, highest_height = original_width, original_height

    if config.HIGH_RES_ENABLED:
        for width, height in config.CAMERA_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width == width and actual_height == height:
                highest_width, highest_height = width, height
                print(f"Set camera resolution to {width}x{height}")
                break

        # If no resolution worked, revert to original
        if highest_width != original_width or highest_height != original_height:
            print(f"Using resolution: {highest_width}x{highest_height}")
        else:
            print(f"Using default resolution: {original_width}x{original_height}")

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your camera connection.")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit the application")
    print(f"Recording videos and taking screenshots of faces with direct eye contact (threshold: {config.EYE_CONTACT_THRESHOLD})")
    print(f"Debounce time between recordings: {config.DEBOUNCE_TIME} seconds")
    print(f"Debounce time between screenshots: {config.SCREENSHOT_DEBOUNCE_TIME} seconds")
    print(f"Video duration: {config.VIDEO_DURATION} seconds")
    print(f"Continue recording after eye contact lost: {config.POST_GAZE_RECORD_TIME} seconds")
    print(f"Face redetection timeout: {config.FACE_REDETECTION_TIMEOUT} seconds")
    print(f"Face margin: {config.FACE_MARGIN_PERCENT}% of original size")
    print(f"Saving videos in MP4 format")
    print(f"High-resolution screenshots: {'Enabled' if config.HIGH_RES_ENABLED else 'Disabled'}")
    print(f"Dynamic configuration window enabled. Use 's' to save, 'l' to load, 'r' to reset settings.")

    # Store current configuration values to detect changes
    current_face_detection_confidence = config.FACE_DETECTION_CONFIDENCE
    current_face_detection_model = config.FACE_DETECTION_MODEL

    # Initialize MediaPipe face detection
    face_detection = mp_face.FaceDetection(
        min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
        model_selection=config.FACE_DETECTION_MODEL
    )

    frame_count = 0
    start_time = time.time()

    # Dictionary to track active face windows
    active_faces = {}

    # Position the main window
    cv2.namedWindow(config.MAIN_WINDOW_NAME)
    cv2.moveWindow(config.MAIN_WINDOW_NAME, *config.MAIN_WINDOW_POSITION)

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

        # Store the original high-resolution frame for screenshots
        high_res_frame = frame.copy()

        # Resize frame for display and processing if needed
        if config.HIGH_RES_ENABLED and (frame.shape[1] > config.DISPLAY_WIDTH or frame.shape[0] > config.DISPLAY_HEIGHT):
            display_frame_original = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
        else:
            display_frame_original = frame.copy()

        # Flip the frames horizontally for a more natural view
        high_res_frame = cv2.flip(high_res_frame, 1)
        display_frame_original = cv2.flip(display_frame_original, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(display_frame_original, cv2.COLOR_BGR2RGB)

        # Check if we need to reinitialize face detection with new parameters
        if (current_face_detection_confidence != config.FACE_DETECTION_CONFIDENCE or
            current_face_detection_model != config.FACE_DETECTION_MODEL):
            # Reinitialize MediaPipe face detection with updated parameters
            face_detection = mp_face.FaceDetection(
                min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
                model_selection=config.FACE_DETECTION_MODEL
            )
            # Update current values
            current_face_detection_confidence = config.FACE_DETECTION_CONFIDENCE
            current_face_detection_model = config.FACE_DETECTION_MODEL
            print(f"Updated face detection parameters: confidence={config.FACE_DETECTION_CONFIDENCE}, model={config.FACE_DETECTION_MODEL}")

        # Detect Faces
        faces = face_detection.process(rgb_frame)

        # Track which faces we've seen in this frame
        current_faces = set()
        current_time = time.time()

        # Create a copy of the frame for drawing indicators
        display_frame = display_frame_original.copy()

        # Calculate scale factors between high-res and display frames
        if config.HIGH_RES_ENABLED:
            scale_x = high_res_frame.shape[1] / display_frame.shape[1]
            scale_y = high_res_frame.shape[0] / display_frame.shape[0]
        else:
            scale_x, scale_y = 1.0, 1.0

        # Process detected faces
        if faces.detections:
            num_faces = len(faces.detections)
            print(f"Detected {num_faces} faces")

            for i, detection in enumerate(faces.detections):
                # Get face ID (using index for now)
                face_id = f"face_{i+1}"
                current_faces.add(face_id)

                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x, y = int(bbox.xmin * display_frame.shape[1]), int(bbox.ymin * display_frame.shape[0])
                w, h = int(bbox.width * display_frame.shape[1]), int(bbox.height * display_frame.shape[0])

                # Calculate margin to add around the face (as a percentage of original size)
                margin_w = int(w * (config.FACE_MARGIN_PERCENT / 100))
                margin_h = int(h * (config.FACE_MARGIN_PERCENT / 100))

                # Expand the bounding box with the margin
                x_expanded = max(0, x - margin_w)
                y_expanded = max(0, y - margin_h)
                w_expanded = min(w + 2 * margin_w, display_frame.shape[1] - x_expanded)
                h_expanded = min(h + 2 * margin_h, display_frame.shape[0] - y_expanded)

                # Store the current face position for tracking when detection fails
                last_face_positions[face_id] = {
                    "x": x_expanded,
                    "y": y_expanded,
                    "w": w_expanded,
                    "h": h_expanded,
                    "last_seen": current_time
                }

                # Extract face region from the display frame for analysis
                if w_expanded > 0 and h_expanded > 0:  # Make sure we have a valid region
                    face_img_display = display_frame_original[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded].copy()

                    # Calculate high-resolution coordinates for the same face region
                    if config.HIGH_RES_ENABLED:
                        x_high_res = int(x_expanded * scale_x)
                        y_high_res = int(y_expanded * scale_y)
                        w_high_res = int(w_expanded * scale_x)
                        h_high_res = int(h_expanded * scale_y)

                        # Make sure the coordinates are within the high-res frame boundaries
                        x_high_res = max(0, x_high_res)
                        y_high_res = max(0, y_high_res)
                        w_high_res = min(w_high_res, high_res_frame.shape[1] - x_high_res)
                        h_high_res = min(h_high_res, high_res_frame.shape[0] - y_high_res)

                        # Extract the high-resolution face region
                        face_img_high_res = high_res_frame[y_high_res:y_high_res+h_high_res, x_high_res:x_high_res+w_high_res].copy()
                    else:
                        face_img_high_res = face_img_display.copy()

                    # Predict eye contact probability using the display resolution face image
                    eye_contact_prob = eye_contact_detector.predict_eye_contact_probability(face_img_display)

                    # Determine if there's eye contact based on a threshold
                    has_eye_contact = eye_contact_prob > config.EYE_CONTACT_THRESHOLD

                    # Create label with eye contact probability
                    eye_contact_label = f"Eye Contact: {eye_contact_prob:.2f}"

                    # Choose color based on eye contact (green for eye contact, red for no eye contact)
                    eye_contact_color = (0, 255, 0) if has_eye_contact else (0, 0, 255)

                    # Handle video recording and screenshot logic for eye contact

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
                        (current_time - eye_contact_tracker[face_id]["last_screenshot_time"] > config.SCREENSHOT_DEBOUNCE_TIME)):

                        # Create a timestamp for the filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_count = eye_contact_tracker[face_id]["screenshot_count"] + 1
                        screenshot_filename = f"{config.SCREENSHOTS_DIR}/face_{i+1}_eye_contact_{timestamp}_{screenshot_count}.jpg"

                        # Save the high-resolution screenshot
                        cv2.imwrite(screenshot_filename, face_img_high_res)

                        # Update tracker
                        eye_contact_tracker[face_id]["last_screenshot_time"] = current_time
                        eye_contact_tracker[face_id]["screenshot_count"] = screenshot_count

                        print(f"Screenshot taken for Face {i+1} with eye contact probability {eye_contact_prob:.2f}")
                        print(f"Screenshot size: {face_img_high_res.shape[1]}x{face_img_high_res.shape[0]}")

                        # Add a screenshot indicator to the label
                        eye_contact_label = f"Screenshot! {eye_contact_label}"

                    # Check if we should start recording a video
                    if (has_eye_contact and
                        not eye_contact_tracker[face_id]["is_recording"] and
                        (current_time - eye_contact_tracker[face_id]["last_recording_time"] > config.DEBOUNCE_TIME)):

                        # Create a timestamp for the filename
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        recording_count = eye_contact_tracker[face_id]["recording_count"] + 1
                        video_filename = f"{config.VIDEOS_DIR}/face_{i+1}_eye_contact_{timestamp}_{recording_count}.mp4"

                        # Use high-resolution dimensions for video if available
                        if config.HIGH_RES_ENABLED:
                            recorder_width = w_high_res
                            recorder_height = h_high_res
                        else:
                            recorder_width = w_expanded
                            recorder_height = h_expanded

                        # Create and start a new recorder
                        recorder = FaceVideoRecorder(
                            filename=video_filename,
                            width=recorder_width,
                            height=recorder_height
                        )
                        recorder.start_recording()
                        active_recorders[face_id] = recorder

                        # Update tracker
                        eye_contact_tracker[face_id]["last_recording_time"] = current_time
                        eye_contact_tracker[face_id]["recording_count"] = recording_count
                        eye_contact_tracker[face_id]["is_recording"] = True
                        eye_contact_tracker[face_id]["lost_eye_contact_time"] = 0

                        print(f"Started recording video for Face {i+1} with eye contact probability {eye_contact_prob:.2f}")
                        print(f"Video size: {recorder_width}x{recorder_height}")

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
                        # Use high-resolution face image for recording if available
                        if config.HIGH_RES_ENABLED:
                            active_recorders[face_id].add_frame(face_img_high_res)
                        else:
                            active_recorders[face_id].add_frame(face_img_display)

                    # Check if we should stop recording (only if post-gaze time has elapsed)
                    lost_eye_contact_time = eye_contact_tracker[face_id]["lost_eye_contact_time"]
                    if (eye_contact_tracker[face_id]["is_recording"] and
                        lost_eye_contact_time > 0 and
                        current_time - lost_eye_contact_time > config.POST_GAZE_RECORD_TIME and
                        face_id in active_recorders and
                        not active_recorders[face_id].is_recording):

                        eye_contact_tracker[face_id]["is_recording"] = False
                        # Remove the recorder from active recorders
                        if face_id in active_recorders:
                            del active_recorders[face_id]

                    # Update eye contact state
                    eye_contact_tracker[face_id]["last_eye_contact"] = has_eye_contact

                    # Resize face image to standard size for the window
                    face_img_display_resized = cv2.resize(face_img_display, (config.FACE_WINDOW_WIDTH, config.FACE_WINDOW_HEIGHT))

                    # Add eye contact label to the face image
                    cv2.putText(face_img_display_resized, eye_contact_label, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_contact_color, 1)

                    # Add recording indicator if currently recording
                    if eye_contact_tracker[face_id]["is_recording"]:
                        cv2.circle(face_img_display_resized, (config.FACE_WINDOW_WIDTH - 20, 20), 10, (0, 0, 255), -1)  # Red circle indicator

                    # Create or update face window
                    window_name = f"Face {i+1}"

                    # Position the window (main window + offset based on face number)
                    window_x = config.MAIN_WINDOW_POSITION[0] + display_frame.shape[1] + 20  # Main window width + margin
                    window_y = config.MAIN_WINDOW_POSITION[1] + (i * (config.FACE_WINDOW_HEIGHT + 30))  # Vertical offset for each face

                    # Create named window and position it
                    if face_id not in active_faces:
                        cv2.namedWindow(window_name)
                        cv2.moveWindow(window_name, window_x, window_y)
                        active_faces[face_id] = window_name

                    # Display face in its window
                    cv2.imshow(window_name, face_img_display_resized)

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

        # Process faces that weren't detected in this frame but were recently seen
        faces_to_check = set(last_face_positions.keys()) - current_faces
        for face_id in faces_to_check:
            face_data = last_face_positions[face_id]

            # Only process faces that were seen recently (within the timeout period)
            if current_time - face_data["last_seen"] < config.FACE_REDETECTION_TIMEOUT:
                # Extract coordinates
                x_expanded = face_data["x"]
                y_expanded = face_data["y"]
                w_expanded = face_data["w"]
                h_expanded = face_data["h"]

                # Make sure the coordinates are still valid
                if (x_expanded >= 0 and y_expanded >= 0 and
                    x_expanded + w_expanded <= display_frame.shape[1] and
                    y_expanded + h_expanded <= display_frame.shape[0]):

                    # Extract face region using the last known position
                    face_img_display = display_frame_original[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded].copy()

                    # Calculate high-resolution coordinates for the same face region
                    if config.HIGH_RES_ENABLED:
                        x_high_res = int(x_expanded * scale_x)
                        y_high_res = int(y_expanded * scale_y)
                        w_high_res = int(w_expanded * scale_x)
                        h_high_res = int(h_expanded * scale_y)

                        # Make sure the coordinates are within the high-res frame boundaries
                        x_high_res = max(0, x_high_res)
                        y_high_res = max(0, y_high_res)
                        w_high_res = min(w_high_res, high_res_frame.shape[1] - x_high_res)
                        h_high_res = min(h_high_res, high_res_frame.shape[0] - y_high_res)

                        # Extract the high-resolution face region
                        face_img_high_res = high_res_frame[y_high_res:y_high_res+h_high_res, x_high_res:x_high_res+w_high_res].copy()
                    else:
                        face_img_high_res = face_img_display.copy()

                    # Add the frame to any active recorder for this face
                    if face_id in active_recorders and active_recorders[face_id].is_recording:
                        if config.HIGH_RES_ENABLED:
                            active_recorders[face_id].add_frame(face_img_high_res)
                        else:
                            active_recorders[face_id].add_frame(face_img_display)

                        # Draw a dashed rectangle to indicate we're using the last known position
                        # Use a normal line type since LINE_DASHED is not available
                        cv2.rectangle(display_frame, (x_expanded, y_expanded),
                                     (x_expanded + w_expanded, y_expanded + h_expanded), (0, 165, 255), 1)

                        # Add a label to indicate we're still tracking
                        cv2.putText(display_frame, f"Tracking Face {face_id.split('_')[1]}",
                                   (x_expanded, y_expanded - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            else:
                # Face hasn't been seen for too long, remove it from tracking
                if face_id in last_face_positions:
                    del last_face_positions[face_id]

        # Clean up eye contact tracker and recorders for faces that are no longer detected
        faces_to_remove_from_tracker = []
        for face_id in eye_contact_tracker:
            # Only remove faces that haven't been seen for a while and aren't being recorded
            if (face_id not in current_faces and
                (face_id not in last_face_positions or
                 current_time - last_face_positions[face_id]["last_seen"] > config.FACE_REDETECTION_TIMEOUT) and
                (face_id not in active_recorders or not active_recorders[face_id].is_recording)):
                faces_to_remove_from_tracker.append(face_id)

        for face_id in faces_to_remove_from_tracker:
            del eye_contact_tracker[face_id]

        # Close windows for faces that are no longer detected
        faces_to_remove = []
        for face_id in active_faces:
            if (face_id not in current_faces and
                (face_id not in last_face_positions or
                 current_time - last_face_positions[face_id]["last_seen"] > config.FACE_REDETECTION_TIMEOUT)):
                window_name = active_faces[face_id]
                cv2.destroyWindow(window_name)
                faces_to_remove.append(face_id)

        # Remove closed windows from tracking
        for face_id in faces_to_remove:
            del active_faces[face_id]

        # Add instructions text
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add resolution information
        if config.HIGH_RES_ENABLED:
            cv2.putText(display_frame, f"Capture: {high_res_frame.shape[1]}x{high_res_frame.shape[0]}",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Display: {display_frame.shape[1]}x{display_frame.shape[0]}",
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Eye Contact Threshold: {config.EYE_CONTACT_THRESHOLD}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the main frame
        cv2.imshow(config.MAIN_WINDOW_NAME, display_frame)

        # Update configuration window
        config_window.update_window()

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting application...")
            break
        else:
            # Handle configuration window key presses
            config_window.handle_key(key)

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