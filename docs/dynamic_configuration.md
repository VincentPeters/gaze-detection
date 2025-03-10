# Dynamic Configuration with Sliders

## Overview
This feature adds a dynamic configuration window with sliders to adjust face detection and eye contact parameters in real-time. The implementation follows a brutalist/hacker aesthetic and allows users to save and load configuration presets.

## Motivation
The current application has fixed parameters that cannot be adjusted during runtime. This makes it difficult to optimize detection for different environments, lighting conditions, and subjects. By adding dynamic configuration, users can fine-tune parameters in real-time to improve face detection and eye contact recognition.

## Implementation Plan

### 1. Create Configuration Window Module ✅
**File:** `/home/vinnie/gaze-detection/config_window.py`

This new module will:
- Create a separate window with OpenCV trackbars for each configurable parameter ✅
- Implement callback functions to update configuration values in real-time ✅
- Provide save/load functionality for configuration presets ✅
- Include a reset button to restore default values ✅

### 2. Modify Configuration Module ✅
**File:** `/home/vinnie/gaze-detection/config.py`

Minimal changes to:
- Add a mechanism to update configuration values dynamically ✅
- Add functionality to save/load configuration presets ✅
- Ensure thread safety when updating parameters ✅

### 3. Integrate with Main Application ✅
**File:** `/home/vinnie/gaze-detection/main.py`

Minimal changes to:
- Initialize the configuration window ✅
- Check for updated configuration values in the main loop ✅
- Apply updated parameters to face detection and eye contact recognition ✅

## Detailed Tasks

### Task 1: Create Configuration Window Module ✅
1. Create a new file `config_window.py` ✅
2. Implement a `ConfigWindow` class with the following methods: ✅
   - `__init__()`: Initialize the window and trackbars ✅
   - `create_trackbars()`: Create trackbars for each parameter ✅
   - `update_config()`: Update configuration values based on trackbar positions ✅
   - `save_config()`: Save current configuration to a file ✅
   - `load_config()`: Load configuration from a file ✅
   - `reset_config()`: Reset configuration to default values ✅
3. Implement callback functions for each trackbar ✅
4. Create a brutalist/hacker-style UI with minimal decoration ✅

### Task 2: Modify Configuration Module ✅
1. Add dynamic update capability to `config.py` ✅
2. Implement functions to save/load configuration presets ✅
3. Ensure thread safety when updating parameters ✅
4. Maintain backward compatibility with existing code ✅

### Task 3: Integrate with Main Application ✅
1. Import the `ConfigWindow` class in `main.py` ✅
2. Initialize the configuration window in the `main()` function ✅
3. Add code to check for updated configuration values in the main loop ✅
4. Apply updated parameters to face detection and eye contact recognition ✅

## Parameters Included

### Face Detection Parameters
- `FACE_DETECTION_CONFIDENCE` (0.1-1.0): Minimum confidence for face detection ✅
- `FACE_DETECTION_MODEL` (0-1): MediaPipe model selection (0 for close-range, 1 for full-range) ✅
- `FACE_MARGIN_PERCENT` (0-100): Percentage of margin to add around detected faces ✅
- `FACE_REDETECTION_TIMEOUT` (0.1-5.0): How long to keep tracking a face after detection fails ✅

### Eye Contact Detection Parameters
- `EYE_CONTACT_THRESHOLD` (0.1-1.0): Threshold for determining eye contact ✅
- `DEBOUNCE_TIME` (0.1-10.0): Seconds between video recordings for the same face ✅
- `SCREENSHOT_DEBOUNCE_TIME` (0.1-5.0): Seconds between screenshots for the same face ✅
- `POST_GAZE_RECORD_TIME` (0.1-5.0): Continue recording for this many seconds after eye contact is lost ✅

### Camera/Video Settings
- `HIGH_RES_ENABLED` (On/Off): Enable high-resolution capture ✅
- `VIDEO_FPS` (5-60): Frames per second for recorded videos ✅

## Configuration Preset Management
- Save current configuration to a JSON file ✅
- Load configuration from a JSON file ✅
- Reset to default configuration ✅
- Display current preset name ✅

## UI Design
- Brutalist/hacker aesthetic with minimal decoration ✅
- Clear labels and value displays for each parameter ✅
- Group related parameters together ✅
- High contrast colors for better visibility ✅
- Keyboard shortcuts for common actions ✅

## Performance Considerations
- The implementation has minimal impact on performance ✅
- Configuration window updates are throttled to avoid excessive CPU usage ✅
- Configuration changes are applied efficiently ✅

## Testing Plan
1. Test each parameter individually to ensure it affects the application as expected
2. Test saving and loading configuration presets
3. Test performance impact of the configuration window
4. Test on Raspberry Pi to ensure compatibility and performance

## Future Enhancements
- Add visual feedback for parameter changes (e.g., highlighting changed values)
- Add tooltips or help text explaining what each parameter does
- Add ability to compare different configurations side-by-side
- Add automatic parameter optimization based on detection results

## Usage Instructions
1. Run the application as usual
2. A "Detection Settings" window will appear alongside the main window
3. Adjust sliders to change detection parameters in real-time
4. Press 'S' to save the current configuration
5. Press 'L' to load a saved configuration
6. Press 'R' to reset to default values
7. Press 'Q' in the main window to quit the application