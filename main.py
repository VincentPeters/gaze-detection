# Face Tracking with Individual Face Windows
import cv2
import mediapipe as mp
import os
import time
import numpy as np
from datetime import datetime
import threading
import queue
import config  # Import the configuration file
from config_window import ConfigWindow  # Import the configuration window
import tkinter as tk  # Import tkinter for the new layout
from log_redirect import LogRedirector

# Import layout manager if enabled in config
if config.USE_TKINTER_LAYOUT:
    from layout_manager import LayoutManager

# Import eye contact detection model
from eye_contact_model import EyeContactDetector

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
        self.frames = []
        self.recording = False
        self.writer = None
        self.thread = None

    def start_recording(self):
        self.recording = True
        self.frames = []
        print(f"Started recording to {self.filename}")

    def add_frame(self, frame):
        if self.recording:
            # Resize frame if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.frames.append(frame.copy())

    def add_last_frame_if_available(self):
        if self.frames and self.recording:
            self.frames.append(self.frames[-1].copy())

    def _record_video(self):
        # Create video writer with MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))

        # Write all frames
        for frame in self.frames:
            self.writer.write(frame)

        # Release the writer
        self.writer.release()
        print(f"Finished recording {self.filename}")

        # Reset recording state
        self.recording = False
        self.frames = []

    def stop_recording(self):
        if self.recording:
            self.recording = False
            # Start a new thread to write the video
            self.thread = threading.Thread(target=self._record_video)
            self.thread.daemon = True
            self.thread.start()

    def is_recording(self):
        return self.recording

class FaceTrackingApp:
    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(config.FACES_DIR, exist_ok=True)
        os.makedirs(config.VIDEOS_DIR, exist_ok=True)
        os.makedirs(config.SCREENSHOTS_DIR, exist_ok=True)

        # Initialize Tkinter root if using Tkinter layout
        self.tk_root = None
        self.layout_manager = None
        self.log_redirector = None

        if config.USE_TKINTER_LAYOUT:
            self.tk_root = tk.Tk()
            self.tk_root.title("Face Tracking with Eye Contact Detection")

            # Set initial window size (80% of screen)
            screen_width = self.tk_root.winfo_screenwidth()
            screen_height = self.tk_root.winfo_screenheight()
            width = int(screen_width * 0.8)
            height = int(screen_height * 0.8)
            self.tk_root.geometry(f"{width}x{height}")

            # Initialize layout manager
            self.layout_manager = LayoutManager(root=self.tk_root, enable_fullscreen=config.ENABLE_FULLSCREEN)

            # Set up log redirector
            self.log_redirector = LogRedirector(self.layout_manager.log_text)
            self.log_redirector.start_redirect()

            # Process initial events to ensure widgets are properly initialized
            self.tk_root.update_idletasks()

        # Print configuration information
        print(f"Using device: CPU")
        print("Starting camera capture...")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use default camera
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Camera opened successfully!")
        print("Press 'q' to quit the application")

        # Set camera properties if specified
        if hasattr(config, 'CAMERA_WIDTH') and hasattr(config, 'CAMERA_HEIGHT') and config.CAMERA_WIDTH and config.CAMERA_HEIGHT:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        # Print recording configuration
        print(f"Recording videos and taking screenshots of faces with direct eye contact (threshold: {config.EYE_CONTACT_THRESHOLD})")
        print(f"Video recording: {'Enabled' if config.VIDEO_CAPTURE_ENABLED else 'Disabled'}")
        print(f"Screenshot capturing: {'Enabled' if config.IMAGE_CAPTURE_ENABLED else 'Disabled'}")
        print(f"Debounce time between recordings: {config.DEBOUNCE_TIME} seconds")
        print(f"Debounce time between screenshots: {config.SCREENSHOT_DEBOUNCE_TIME} seconds")
        print(f"Video duration: {config.VIDEO_DURATION} seconds")
        print(f"Continue recording after eye contact lost: {config.POST_GAZE_RECORD_TIME} seconds")
        print(f"Face redetection timeout: {config.FACE_REDETECTION_TIMEOUT} seconds")
        print(f"Face margin: {config.FACE_MARGIN_PERCENT}% of original size")
        print(f"Saving videos in MP4 format")
        print(f"High-resolution screenshots: {'Enabled' if config.HIGH_RES_ENABLED else 'Disabled'}")
        print(f"Dynamic configuration window enabled. Press 'c' to toggle configuration window, 'r' to reset settings.")

        if self.layout_manager:
            print(f"Using Tkinter layout: {'Enabled' if config.USE_TKINTER_LAYOUT else 'Disabled'}")
            print(f"Fullscreen mode: {'Enabled' if config.ENABLE_FULLSCREEN else 'Disabled'}")
            print(f"Layout theme: {config.LAYOUT_THEME}")

        # Initialize MediaPipe face detection
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            model_selection=config.FACE_DETECTION_MODEL
        )

        self.frame_count = 0
        self.start_time = time.time()

        # Dictionary to track active face windows
        self.active_faces = {}

        # Position the main window (only if not using Tkinter layout)
        if not self.layout_manager:
            cv2.namedWindow(config.MAIN_WINDOW_NAME)
            cv2.moveWindow(config.MAIN_WINDOW_NAME, *config.MAIN_WINDOW_POSITION)

        # Initialize eye contact detector
        self.eye_contact_detector = EyeContactDetector(model_path=config.MODEL_PATH)

        # Dictionary to track last recording time for each face
        self.last_recording_time = {}
        self.last_screenshot_time = {}

        # Dictionary to track eye contact status
        self.eye_contact_status = {}
        self.eye_contact_start_time = {}

        # Dictionary to track face video recorders
        self.face_recorders = {}

        # Dictionary to track when faces were last seen
        self.last_seen_time = {}

        # Flag to indicate if the application should quit
        self.should_quit = False

        # Bind keyboard shortcuts if using Tkinter
        if self.tk_root:
            self.tk_root.bind('<q>', self.quit_app)
            self.tk_root.bind('<c>', self.toggle_config_window)
            self.tk_root.bind('<r>', self.reset_config)

            # Schedule the first frame processing
            self.tk_root.after(10, self.process_frame)

    def quit_app(self, event=None):
        """Quit the application."""
        self.should_quit = True
        if self.tk_root:
            self.tk_root.quit()

    def toggle_config_window(self, event=None):
        """Toggle the configuration window."""
        if hasattr(config, 'ENABLE_CONFIG_WINDOW') and config.ENABLE_CONFIG_WINDOW:
            # Import here to avoid circular import
            from config_window import show_config_window
            show_config_window()

    def reset_config(self, event=None):
        """Reset configuration to defaults."""
        if hasattr(config, 'ENABLE_CONFIG_WINDOW') and config.ENABLE_CONFIG_WINDOW:
            config.reset_config()
            print("Configuration reset to defaults")

    def process_frame(self):
        """Process a single frame from the camera."""
        if self.should_quit:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame - camera disconnected or end of video file")
            self.quit_app()
            return

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if self.frame_count % 30 == 0:  # Print every 30 frames
            fps = self.frame_count / elapsed_time
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

        # Create a copy for drawing
        display_frame = display_frame_original.copy()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self.face_detection.process(rgb_frame)

        # Set of currently detected face IDs
        current_faces = set()

        # Process face detections
        if results.detections:
            print(f"Detected {len(results.detections)} faces")

            for i, detection in enumerate(results.detections):
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = display_frame.shape

                # Convert relative coordinates to absolute
                xmin = max(0, int(bbox.xmin * iw))
                ymin = max(0, int(bbox.ymin * ih))
                width = min(int(bbox.width * iw), iw - xmin)
                height = min(int(bbox.height * ih), ih - ymin)

                # Add margin to the face crop
                margin_x = int(width * config.FACE_MARGIN_PERCENT / 100)
                margin_y = int(height * config.FACE_MARGIN_PERCENT / 100)

                # Calculate expanded bounding box with margin
                xmin_expanded = max(0, xmin - margin_x)
                ymin_expanded = max(0, ymin - margin_y)
                width_expanded = min(width + 2 * margin_x, iw - xmin_expanded)
                height_expanded = min(height + 2 * margin_y, ih - ymin_expanded)

                # Extract face with margin
                face_img = display_frame[ymin_expanded:ymin_expanded+height_expanded,
                                        xmin_expanded:xmin_expanded+width_expanded]

                # Skip if face extraction failed
                if face_img.size == 0:
                    continue

                # Generate a unique ID for this face based on its position
                face_id = f"face_{i+1}"
                current_faces.add(face_id)

                # Update last seen time for this face
                self.last_seen_time[face_id] = time.time()

                # Draw bounding box on the display frame
                cv2.rectangle(display_frame, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)

                # Add face ID text
                cv2.putText(display_frame, face_id, (xmin, ymin - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Detect eye contact
                eye_contact_score = self.eye_contact_detector.predict_eye_contact_probability(face_img)
                has_eye_contact = eye_contact_score > config.EYE_CONTACT_THRESHOLD

                # Draw eye contact status
                status_text = f"Eye contact: {eye_contact_score:.2f}"
                cv2.putText(display_frame, status_text, (xmin, ymin + height + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0) if has_eye_contact else (0, 0, 255), 1)

                # Track eye contact status changes
                if face_id not in self.eye_contact_status:
                    self.eye_contact_status[face_id] = False

                # Check if eye contact status changed
                if has_eye_contact and not self.eye_contact_status[face_id]:
                    # Eye contact started
                    self.eye_contact_status[face_id] = True
                    self.eye_contact_start_time[face_id] = time.time()
                    print(f"Eye contact detected for {face_id}! Score: {eye_contact_score:.2f}")

                    # Start recording if enabled and not in debounce period
                    current_time = time.time()
                    if (config.VIDEO_CAPTURE_ENABLED and
                        (face_id not in self.last_recording_time or
                         current_time - self.last_recording_time.get(face_id, 0) > config.DEBOUNCE_TIME)):

                        # Create a timestamp for the filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_filename = os.path.join(config.VIDEOS_DIR, f"{face_id}_{timestamp}.mp4")

                        # Create a recorder for this face if it doesn't exist
                        if face_id not in self.face_recorders:
                            self.face_recorders[face_id] = FaceVideoRecorder(
                                video_filename,
                                face_img.shape[1],
                                face_img.shape[0],
                                fps=config.VIDEO_FPS,
                                duration=config.VIDEO_DURATION
                            )

                        # Start recording
                        self.face_recorders[face_id].start_recording()
                        self.last_recording_time[face_id] = current_time

                    # Take screenshot if enabled and not in debounce period
                    if (config.IMAGE_CAPTURE_ENABLED and
                        (face_id not in self.last_screenshot_time or
                         current_time - self.last_screenshot_time.get(face_id, 0) > config.SCREENSHOT_DEBOUNCE_TIME)):

                        # Create a timestamp for the filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_filename = os.path.join(config.SCREENSHOTS_DIR, f"{face_id}_{timestamp}.jpg")

                        # Use high-res frame if enabled
                        if config.HIGH_RES_ENABLED:
                            # Calculate coordinates in high-res frame
                            scale_x = high_res_frame.shape[1] / display_frame.shape[1]
                            scale_y = high_res_frame.shape[0] / display_frame.shape[0]

                            hr_xmin = int(xmin_expanded * scale_x)
                            hr_ymin = int(ymin_expanded * scale_y)
                            hr_width = int(width_expanded * scale_x)
                            hr_height = int(height_expanded * scale_y)

                            # Extract face from high-res frame
                            hr_face_img = high_res_frame[hr_ymin:hr_ymin+hr_height, hr_xmin:hr_xmin+hr_width]
                            cv2.imwrite(screenshot_filename, hr_face_img)
                        else:
                            # Use display resolution
                            cv2.imwrite(screenshot_filename, face_img)

                        print(f"Saved screenshot to {screenshot_filename}")
                        self.last_screenshot_time[face_id] = current_time

                elif not has_eye_contact and self.eye_contact_status[face_id]:
                    # Eye contact ended
                    self.eye_contact_status[face_id] = False
                    print(f"Eye contact lost for {face_id}. Score: {eye_contact_score:.2f}")

                    # If recording, continue for a short time then stop
                    if face_id in self.face_recorders and self.face_recorders[face_id].is_recording():
                        def stop_recording_later(face_id):
                            time.sleep(config.POST_GAZE_RECORD_TIME)
                            if face_id in self.face_recorders:
                                # Add a few more frames of the last frame
                                for _ in range(5):
                                    self.face_recorders[face_id].add_last_frame_if_available()
                                self.face_recorders[face_id].stop_recording()

                        # Start a thread to stop recording after delay
                        stop_thread = threading.Thread(
                            target=stop_recording_later,
                            args=(face_id,)
                        )
                        stop_thread.daemon = True
                        stop_thread.start()

                # Add frame to recorder if recording
                if face_id in self.face_recorders and self.face_recorders[face_id].is_recording():
                    self.face_recorders[face_id].add_frame(face_img)

                # Display face in a separate window or panel
                is_recording = face_id in self.face_recorders and self.face_recorders[face_id].is_recording()

                # If using Tkinter layout, update the face panel
                if self.layout_manager:
                    # Use the face index (0-3) for the panel
                    face_index = min(i, 3)  # Limit to 4 panels (0-3)
                    self.layout_manager.update_face_panel(face_index, face_img, is_recording)
                else:
                    # Create or update OpenCV window for this face
                    window_name = f"Face {i+1}"
                    if window_name not in self.active_faces:
                        cv2.namedWindow(window_name)
                        # Position windows in a grid
                        row, col = divmod(len(self.active_faces), 2)
                        x_pos = config.MAIN_WINDOW_POSITION[0] + config.DISPLAY_WIDTH + 30 + col * 220
                        y_pos = config.MAIN_WINDOW_POSITION[1] + row * 220
                        cv2.moveWindow(window_name, x_pos, y_pos)
                        self.active_faces[window_name] = face_id

                    # Add recording indicator if recording
                    if is_recording:
                        # Draw red circle in top-right corner
                        circle_radius = 10
                        cv2.circle(face_img, (face_img.shape[1] - circle_radius - 10, circle_radius + 10),
                                  circle_radius, (0, 0, 255), -1)

                    # Show the face
                    cv2.imshow(window_name, face_img)

        # Check for faces that are no longer detected
        faces_to_remove = []
        for window_name, face_id in self.active_faces.items():
            if face_id not in current_faces:
                # Check if face has been gone long enough to close the window
                if (face_id in self.last_seen_time and
                    time.time() - self.last_seen_time[face_id] > config.FACE_REDETECTION_TIMEOUT):
                    # Close the window if not using Tkinter
                    if not self.layout_manager:
                        cv2.destroyWindow(window_name)
                    faces_to_remove.append(window_name)

                    # Stop any ongoing recording
                    if face_id in self.face_recorders and self.face_recorders[face_id].is_recording():
                        self.face_recorders[face_id].stop_recording()

        # Remove closed windows from tracking
        for window_name in faces_to_remove:
            face_id = self.active_faces[window_name]
            del self.active_faces[window_name]

            # Clear the face panel in Tkinter layout
            if self.layout_manager:
                # Find the panel index for this face
                for i in range(4):
                    if f"face_{i+1}" == face_id:
                        self.layout_manager.clear_face_panel(i)
                        break

        # Display the main frame
        if self.layout_manager:
            # Update the camera feed in Tkinter
            self.layout_manager.update_camera_feed(display_frame)
        else:
            # Show in OpenCV window
            cv2.imshow(config.MAIN_WINDOW_NAME, display_frame)

            # Check for key press in OpenCV window
            key = cv2.waitKey(1) & 0xFF

            # 'q' to quit
            if key == ord('q'):
                self.quit_app()
                return

            # 'c' to toggle configuration window
            if key == ord('c') and hasattr(config, 'ENABLE_CONFIG_WINDOW') and config.ENABLE_CONFIG_WINDOW:
                self.toggle_config_window()

            # 'r' to reset configuration
            if key == ord('r') and hasattr(config, 'ENABLE_CONFIG_WINDOW') and config.ENABLE_CONFIG_WINDOW:
                self.reset_config()

        # Schedule the next frame processing if using Tkinter
        if self.tk_root and not self.should_quit:
            self.tk_root.after(10, self.process_frame)

    def run(self):
        """Run the application."""
        if self.tk_root:
            # Start the Tkinter main loop
            self.tk_root.mainloop()
        else:
            # Run the OpenCV main loop
            while not self.should_quit:
                self.process_frame()

                # Break if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        # Clean up
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        # Release the camera
        self.cap.release()

        # Close all OpenCV windows if not using Tkinter
        if not self.layout_manager:
            cv2.destroyAllWindows()
        else:
            # Stop log redirection
            if self.log_redirector:
                self.log_redirector.stop_redirect()

        # Stop any ongoing recordings
        for face_id, recorder in self.face_recorders.items():
            if recorder.is_recording():
                recorder.stop_recording()

        print("Application closed")

def main():
    app = FaceTrackingApp()
    app.run()

if __name__ == "__main__":
    main()