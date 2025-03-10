# Dynamic Configuration Technical Specification

## Overview

This document provides detailed technical specifications for implementing dynamic configuration with sliders in the gaze detection application. The implementation follows a minimally invasive approach to preserve existing functionality while adding new features.

## Architecture ✅

The implementation follows a modular architecture with three main components:

1. **Configuration Module (`config.py`)**: Stores configuration values and provides functions to update them dynamically. ✅
2. **Configuration Window Module (`config_window.py`)**: Creates and manages the UI for adjusting configuration values. ✅
3. **Main Application (`main.py`)**: Uses the configuration values and initializes the configuration window. ✅

## Component Specifications

### 1. Configuration Module (`config.py`) ✅

#### Current State
Currently, `config.py` contains static configuration variables used throughout the application.

#### Changes Required
- Add a dictionary to store default values ✅
- Add functions to update configuration values dynamically ✅
- Add functions to save/load configuration presets ✅
- Add thread safety mechanisms ✅

#### Detailed Specifications

```python
# Add at the end of config.py

import json
import threading
import os

# Directory for configuration presets
CONFIG_PRESETS_DIR = "config_presets"
os.makedirs(CONFIG_PRESETS_DIR, exist_ok=True)

# Store default values
DEFAULT_CONFIG = {
    'FACE_DETECTION_CONFIDENCE': FACE_DETECTION_CONFIDENCE,
    'FACE_DETECTION_MODEL': FACE_DETECTION_MODEL,
    'FACE_MARGIN_PERCENT': FACE_MARGIN_PERCENT,
    'FACE_REDETECTION_TIMEOUT': FACE_REDETECTION_TIMEOUT,
    'EYE_CONTACT_THRESHOLD': EYE_CONTACT_THRESHOLD,
    'DEBOUNCE_TIME': DEBOUNCE_TIME,
    'SCREENSHOT_DEBOUNCE_TIME': SCREENSHOT_DEBOUNCE_TIME,
    'POST_GAZE_RECORD_TIME': POST_GAZE_RECORD_TIME,
    'HIGH_RES_ENABLED': HIGH_RES_ENABLED,
    'VIDEO_FPS': VIDEO_FPS
}

# Thread lock for config updates
config_lock = threading.Lock()

def update_config(param_name, value):
    """Update a configuration parameter with thread safety."""
    with config_lock:
        globals()[param_name] = value
    return value

def get_current_config():
    """Get the current configuration as a dictionary."""
    current_config = {}
    for param_name in DEFAULT_CONFIG.keys():
        current_config[param_name] = globals()[param_name]
    return current_config

def save_config(preset_name):
    """Save the current configuration as a preset."""
    if not preset_name.endswith('.json'):
        preset_name += '.json'

    filepath = os.path.join(CONFIG_PRESETS_DIR, preset_name)
    with open(filepath, 'w') as f:
        json.dump(get_current_config(), f, indent=4)
    return filepath

def load_config(preset_name):
    """Load a configuration preset."""
    if not preset_name.endswith('.json'):
        preset_name += '.json'

    filepath = os.path.join(CONFIG_PRESETS_DIR, preset_name)
    try:
        with open(filepath, 'r') as f:
            preset_config = json.load(f)

        # Update all parameters
        with config_lock:
            for param_name, value in preset_config.items():
                if param_name in globals():
                    globals()[param_name] = value

        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False

def reset_config():
    """Reset all configuration parameters to their default values."""
    with config_lock:
        for param_name, value in DEFAULT_CONFIG.items():
            globals()[param_name] = value
    return DEFAULT_CONFIG

def get_config_presets():
    """Get a list of available configuration presets."""
    presets = []
    for filename in os.listdir(CONFIG_PRESETS_DIR):
        if filename.endswith('.json'):
            presets.append(filename)
    return presets
```

### 2. Configuration Window Module (`config_window.py`) ✅

#### Purpose
Create a new module to handle the UI for adjusting configuration values.

#### Detailed Specifications

```python
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
        self.window_height = 600
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

    def reset_config(self):
        """Reset all configuration parameters to their default values."""
        config.reset_config()
        self.current_preset = "default"
        self.update_trackbars()
        self.update_window()

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
```

### 3. Main Application Integration (`main.py`) ✅

#### Current State
Currently, `main.py` uses static configuration values from `config.py`.

#### Changes Required
- Import and initialize the `ConfigWindow` class ✅
- Check for updated configuration values in the main loop ✅
- Apply updated parameters to face detection and eye contact recognition ✅

#### Detailed Specifications

```python
# Add import at the top of main.py
from config_window import ConfigWindow

# Modify main() function to initialize the configuration window
def main():
    # Existing code...

    # Initialize configuration window
    config_window = ConfigWindow()

    # In the main loop, after checking for 'q' key press:
    # Check for configuration window key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting application...")
        break
    else:
        config_window.handle_key(key)

    # Update configuration window
    config_window.update_window()

    # When reinitializing face detection in the loop:
    face_detection = mp_face.FaceDetection(
        min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
        model_selection=config.FACE_DETECTION_MODEL
    )

    # Rest of the existing code...
```

## Implementation Sequence ✅

1. **Phase 1: Configuration Module Enhancement** ✅
   - Modify `config.py` to add dynamic update capability ✅
   - Add save/load functionality ✅
   - Add thread safety mechanisms ✅
   - Test basic functionality ✅

2. **Phase 2: Configuration Window Implementation** ✅
   - Create `config_window.py` ✅
   - Implement trackbars and callbacks ✅
   - Implement save/load UI ✅
   - Test window functionality in isolation ✅

3. **Phase 3: Main Application Integration** ✅
   - Modify `main.py` to initialize the configuration window ✅
   - Integrate key handling ✅
   - Apply updated parameters in the main loop ✅
   - Test full integration ✅

## Testing Plan ✅

### Unit Testing ✅
- Test configuration update functions ✅
- Test save/load functionality ✅
- Test trackbar callbacks ✅

### Integration Testing ✅
- Test configuration window with main application ✅
- Test parameter updates during runtime ✅
- Test save/load functionality during runtime ✅

### Performance Testing ✅
- Measure CPU and memory usage with configuration window open ✅
- Compare frame rate with and without configuration window ✅
- Test on Raspberry Pi to verify performance ✅

## Deployment Checklist ✅

- [x] Implement changes to `config.py`
- [x] Create `config_window.py`
- [x] Modify `main.py`
- [x] Create `config_presets` directory
- [x] Test all functionality
- [x] Update documentation
- [x] Deploy to Raspberry Pi
- [x] Verify functionality on Raspberry Pi

## Fallback Plan

If performance issues arise:
- Reduce update frequency of the configuration window
- Simplify UI elements
- Consider alternative approaches for problematic components

If integration issues arise:
- Isolate the problematic component
- Implement a simplified version
- Ensure core functionality remains unaffected

## Conclusion

The dynamic configuration feature has been successfully implemented, allowing users to adjust face detection and eye contact parameters in real-time. The implementation follows a minimally invasive approach, preserving existing functionality while adding new features. The brutalist/hacker aesthetic provides a clean and functional UI for adjusting parameters.