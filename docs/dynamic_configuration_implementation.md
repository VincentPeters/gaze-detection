# Dynamic Configuration Implementation Plan

This document outlines the specific implementation details for adding dynamic configuration with sliders to the gaze detection application.

## File Changes

### 1. Create New File: `config_window.py` ✅

This file will contain the `ConfigWindow` class responsible for creating and managing the configuration window with sliders.

Key components:
- Window initialization with OpenCV ✅
- Trackbar creation for each parameter ✅
- Callback functions for trackbar changes ✅
- Configuration save/load functionality ✅
- Reset functionality ✅

### 2. Modify Existing File: `config.py` ✅

Current state: Contains static configuration variables.
Changes needed:
- Add a mechanism to update configuration values dynamically ✅
- Add functionality to save/load configuration presets ✅
- Ensure thread safety when updating parameters ✅

### 3. Modify Existing File: `main.py` ✅

Current state: Uses static configuration values from `config.py`.
Changes needed:
- Import and initialize the `ConfigWindow` class ✅
- Check for updated configuration values in the main loop ✅
- Apply updated parameters to face detection and eye contact recognition ✅

## Detailed Implementation Steps

### Step 1: Modify `config.py` ✅

1. Add a dictionary to store default values for all configurable parameters ✅
2. Add functions to update configuration values dynamically ✅
3. Add functions to save/load configuration presets ✅
4. Add thread safety mechanisms ✅

```python
# Pseudocode for changes to config.py
# Add at the end of the file:

import json
import threading
import os

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

# Functions for dynamic updates
def update_config(param_name, value):
    with config_lock:
        globals()[param_name] = value

def save_config(filename):
    # Implementation details

def load_config(filename):
    # Implementation details

def reset_config():
    # Implementation details
```

### Step 2: Create `config_window.py` ✅

1. Create the `ConfigWindow` class ✅
2. Implement trackbars for each parameter ✅
3. Implement callback functions ✅
4. Implement save/load functionality ✅
5. Implement reset functionality ✅

```python
# Pseudocode for config_window.py
import cv2
import config
import json
import os

class ConfigWindow:
    def __init__(self):
        # Implementation details

    def create_trackbars(self):
        # Implementation details

    def update_config(self):
        # Implementation details

    def save_config(self):
        # Implementation details

    def load_config(self):
        # Implementation details

    def reset_config(self):
        # Implementation details
```

### Step 3: Modify `main.py` ✅

1. Import the `ConfigWindow` class ✅
2. Initialize the configuration window in the `main()` function ✅
3. Check for updated configuration values in the main loop ✅
4. Apply updated parameters to face detection and eye contact recognition ✅

```python
# Pseudocode for changes to main.py
# Add import at the top:
from config_window import ConfigWindow

# Modify main() function:
def main():
    # Existing code...

    # Initialize configuration window
    config_window = ConfigWindow()

    # In the main loop, after processing frames:
    # Apply updated parameters to face detection
    face_detection = mp_face.FaceDetection(
        min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
        model_selection=config.FACE_DETECTION_MODEL
    )

    # Rest of the existing code...
```

## Testing Strategy ✅

1. Test each parameter individually: ✅
   - Adjust face detection confidence and verify detection changes ✅
   - Adjust eye contact threshold and verify recognition changes ✅
   - Test other parameters similarly ✅

2. Test configuration presets: ✅
   - Save current configuration ✅
   - Change parameters ✅
   - Load saved configuration ✅
   - Verify parameters are restored correctly ✅

3. Test performance: ✅
   - Monitor CPU and memory usage with configuration window open ✅
   - Ensure minimal impact on frame rate ✅
   - Test on Raspberry Pi to verify performance ✅

## Deployment Steps ✅

1. Implement changes to `config.py` ✅
2. Create `config_window.py` ✅
3. Modify `main.py` ✅
4. Test all functionality ✅
5. Create a directory for configuration presets ✅
6. Update documentation ✅

## Fallback Plan

If any issues arise during implementation:
1. Isolate the problematic component
2. Implement a simplified version if necessary
3. Consider alternative approaches for problematic components
4. Ensure the core functionality (face detection and eye contact recognition) remains unaffected