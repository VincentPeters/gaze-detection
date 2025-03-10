import cv2
import numpy as np
import config
import os
import json

class ConfigWindow:
    def __init__(self):
        """Initialize the configuration window with trackbars."""
        self.window_name = "Detection Settings"
        self.window_width = 400
        self.window_height = 700  # Increased height for more sliders
        self.background_color = (30, 30, 30)  # Dark gray background for brutalist style
        self.text_color = (200, 200, 200)     # Light gray text
        self.highlight_color = (0, 255, 0)    # Green highlights

        # Create window
        cv2.namedWindow(self.window_name)
        cv2.moveWindow(self.window_name, 50, 50)

        # Create trackbars
        self.create_trackbars()

        # Current preset name
        self.current_preset = "default"

        # Create background image
        self.background = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        self.background[:] = self.background_color

        # Update window
        self.update_window()

    def create_trackbars(self):
        """Create trackbars for each configurable parameter."""
        # Face Detection Parameters
        cv2.createTrackbar("Face Confidence", self.window_name,
                          int(config.FACE_DETECTION_CONFIDENCE * 100), 100,
                          lambda x: config.update_config('FACE_DETECTION_CONFIDENCE', x / 100))

        cv2.createTrackbar("Face Model", self.window_name,
                          config.FACE_DETECTION_MODEL, 1,
                          lambda x: config.update_config('FACE_DETECTION_MODEL', x))

        cv2.createTrackbar("Face Margin %", self.window_name,
                          config.FACE_MARGIN_PERCENT, 100,
                          lambda x: config.update_config('FACE_MARGIN_PERCENT', x))

        cv2.createTrackbar("Redetection Time", self.window_name,
                          int(config.FACE_REDETECTION_TIMEOUT * 10), 50,
                          lambda x: config.update_config('FACE_REDETECTION_TIMEOUT', x / 10))

        # Eye Contact Parameters
        cv2.createTrackbar("Eye Contact Threshold", self.window_name,
                          int(config.EYE_CONTACT_THRESHOLD * 100), 100,
                          lambda x: config.update_config('EYE_CONTACT_THRESHOLD', x / 100))

        cv2.createTrackbar("Debounce Time", self.window_name,
                          int(config.DEBOUNCE_TIME * 10), 100,
                          lambda x: config.update_config('DEBOUNCE_TIME', x / 10))

        cv2.createTrackbar("Screenshot Debounce", self.window_name,
                          int(config.SCREENSHOT_DEBOUNCE_TIME * 10), 50,
                          lambda x: config.update_config('SCREENSHOT_DEBOUNCE_TIME', x / 10))

        cv2.createTrackbar("Post Gaze Record", self.window_name,
                          int(config.POST_GAZE_RECORD_TIME * 10), 50,
                          lambda x: config.update_config('POST_GAZE_RECORD_TIME', x / 10))

        # Video Settings
        cv2.createTrackbar("High Res", self.window_name,
                          1 if config.HIGH_RES_ENABLED else 0, 1,
                          lambda x: config.update_config('HIGH_RES_ENABLED', bool(x)))

        cv2.createTrackbar("Video FPS", self.window_name,
                          config.VIDEO_FPS, 60,
                          lambda x: config.update_config('VIDEO_FPS', max(5, x)))

        # Performance Settings
        cv2.createTrackbar("Process Width", self.window_name,
                          config.PROCESSING_WIDTH, 640,
                          lambda x: config.update_config('PROCESSING_WIDTH', max(160, x)))

        cv2.createTrackbar("Process Height", self.window_name,
                          config.PROCESSING_HEIGHT, 480,
                          lambda x: config.update_config('PROCESSING_HEIGHT', max(120, x)))

        cv2.createTrackbar("Frame Interval", self.window_name,
                          config.FRAME_PROCESSING_INTERVAL, 10,
                          lambda x: config.update_config('FRAME_PROCESSING_INTERVAL', max(1, x)))

    def update_window(self):
        """Update the configuration window with current values."""
        # Create a copy of the background
        display = self.background.copy()

        # Add title
        cv2.putText(display, "DETECTION SETTINGS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.highlight_color, 2)

        # Add current preset name
        cv2.putText(display, f"Preset: {self.current_preset}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 1)

        # Add section titles
        cv2.putText(display, "PERFORMANCE SETTINGS", (10, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.highlight_color, 1)

        cv2.putText(display, f"Process Width: {config.PROCESSING_WIDTH}", (10, 430),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        cv2.putText(display, f"Process Height: {config.PROCESSING_HEIGHT}", (10, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        cv2.putText(display, f"Frame Interval: {config.FRAME_PROCESSING_INTERVAL}", (10, 490),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Add instructions
        cv2.putText(display, "S: Save preset", (10, self.window_height - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        cv2.putText(display, "L: Load preset", (10, self.window_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        cv2.putText(display, "R: Reset to defaults", (10, self.window_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        # Display the window
        cv2.imshow(self.window_name, display)

    def handle_key(self, key):
        """Handle keyboard input for the configuration window."""
        if key == ord('s'):
            self.save_config()
        elif key == ord('l'):
            self.load_config()
        elif key == ord('r'):
            self.reset_config()

    def save_config(self):
        """Save the current configuration as a preset."""
        # In a real implementation, this would show a text input dialog
        # For simplicity, we'll use a fixed name based on timestamp
        import time
        preset_name = f"preset_{int(time.time())}.json"
        config.save_config(preset_name)
        self.current_preset = preset_name
        self.update_window()
        print(f"Configuration saved as: {preset_name}")

    def load_config(self):
        """Load a configuration preset."""
        # In a real implementation, this would show a file selection dialog
        # For simplicity, we'll load the first preset found
        presets = config.get_config_presets()
        if presets:
            preset_name = presets[0]
            if config.load_config(preset_name):
                self.current_preset = preset_name
                self.update_trackbars()
                self.update_window()
                print(f"Loaded configuration preset: {preset_name}")
            else:
                print(f"Failed to load configuration preset: {preset_name}")
        else:
            print("No configuration presets found.")

    def reset_config(self):
        """Reset all configuration parameters to their default values."""
        config.reset_config()
        self.current_preset = "default"
        self.update_trackbars()
        self.update_window()
        print("Reset to default configuration.")

    def update_trackbars(self):
        """Update trackbar positions based on current configuration values."""
        cv2.setTrackbarPos("Face Confidence", self.window_name, int(config.FACE_DETECTION_CONFIDENCE * 100))
        cv2.setTrackbarPos("Face Model", self.window_name, config.FACE_DETECTION_MODEL)
        cv2.setTrackbarPos("Face Margin %", self.window_name, config.FACE_MARGIN_PERCENT)
        cv2.setTrackbarPos("Redetection Time", self.window_name, int(config.FACE_REDETECTION_TIMEOUT * 10))
        cv2.setTrackbarPos("Eye Contact Threshold", self.window_name, int(config.EYE_CONTACT_THRESHOLD * 100))
        cv2.setTrackbarPos("Debounce Time", self.window_name, int(config.DEBOUNCE_TIME * 10))
        cv2.setTrackbarPos("Screenshot Debounce", self.window_name, int(config.SCREENSHOT_DEBOUNCE_TIME * 10))
        cv2.setTrackbarPos("Post Gaze Record", self.window_name, int(config.POST_GAZE_RECORD_TIME * 10))
        cv2.setTrackbarPos("High Res", self.window_name, 1 if config.HIGH_RES_ENABLED else 0)
        cv2.setTrackbarPos("Video FPS", self.window_name, config.VIDEO_FPS)
        cv2.setTrackbarPos("Process Width", self.window_name, config.PROCESSING_WIDTH)
        cv2.setTrackbarPos("Process Height", self.window_name, config.PROCESSING_HEIGHT)
        cv2.setTrackbarPos("Frame Interval", self.window_name, config.FRAME_PROCESSING_INTERVAL)