# Configuration file for the eye contact detection application
import json
import threading
import os

# Paths
MODEL_PATH = "models/model_weights.pkl"
VIDEOS_DIR = "eye_contact_videos"
SCREENSHOTS_DIR = "eye_contact_screenshots"
FACES_DIR = "faces"

# UI settings
ENABLE_CONFIG_WINDOW = True  # Enable/disable the configuration window
USE_TKINTER_LAYOUT = True  # Use Tkinter layout instead of OpenCV windows
ENABLE_FULLSCREEN = False  # Start in fullscreen mode

# Streaming settings
ENABLE_STREAMING = True  # Enable/disable streaming server
STREAMING_PORT = 8080    # Port for the streaming server (changed from 5000 to avoid conflicts)
DISABLE_LOCAL_PREVIEW = True  # Disable local preview windows but keep streaming
STREAM_QUALITY = 90      # JPEG quality for streams (0-100)

# Eye contact detection settings
EYE_CONTACT_THRESHOLD = 0.3  # Lowered from 0.5 to be less strict
FACE_MARGIN_PERCENT = 60  # Percentage of margin to add around detected faces

# Capture settings
VIDEO_CAPTURE_ENABLED = False  # Enable/disable video recording
IMAGE_CAPTURE_ENABLED = False  # Enable/disable screenshot capturing

# Timing settings
DEBOUNCE_TIME = 5.0  # Seconds between video recordings for the same face
SCREENSHOT_DEBOUNCE_TIME = 1.0  # Seconds between screenshots for the same face
VIDEO_DURATION = 6.0  # Duration of recorded videos in seconds
POST_GAZE_RECORD_TIME = 1.0  # Continue recording for this many seconds after eye contact is lost
FACE_REDETECTION_TIMEOUT = 1.0  # Time in seconds before closing a face window after detection is lost
FACE_PERSISTENCE_FRAMES = 5  # Number of frames to keep a face visible after detection is lost

# Video recording settings
VIDEO_FPS = 20  # Frames per second for recorded videos

# Resolution settings
HIGH_RES_ENABLED = False  # Enable high-resolution capture
DISPLAY_WIDTH = 640  # Width for display and processing
DISPLAY_HEIGHT = 480  # Height for display and processing

# Camera resolution options to try (in order of preference)
CAMERA_RESOLUTIONS = [
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
    (1024, 768),   # XGA
    (800, 600),    # SVGA
    (640, 480)     # VGA (fallback)
]

# Camera device settings
CAMERA_DEVICE = 0  # Default camera device (0 is usually the built-in webcam)
CAMERA_WIDTH = 1920  # Camera width (0 = default)
CAMERA_HEIGHT = 1080  # Camera height (0 = default)

# Face detection settings
FACE_DETECTION_CONFIDENCE = 0.5  # MediaPipe face detection confidence threshold (0.0-1.0)
FACE_DETECTION_MODEL = 0  # MediaPipe face detection model selection (0: short-range, 1: full-range)
FACE_MARGIN_PERCENT = 60  # Percentage of margin to add around detected faces (%)
FACE_REDETECTION_TIMEOUT = 1.0  # Time in seconds before closing a face window after detection is lost
FACE_PERSISTENCE_FRAMES = 5  # Number of frames to keep a face visible after detection is lost

# Window settingsqs
MAIN_WINDOW_NAME = 'Face Detection'
FACE_WINDOW_WIDTH = 200  # Width of individual face windows
FACE_WINDOW_HEIGHT = 200  # Height of individual face windows
MAIN_WINDOW_POSITION = (50, 50)  # (x, y) position of the main window

# Performance optimization settings
PROCESSING_WIDTH = 640  # Width for face detection processing (smaller = faster)
PROCESSING_HEIGHT = 480  # Height for face detection processing (smaller = faster)
FRAME_PROCESSING_INTERVAL = 1  # Process every n frames (can improve performance)
DISPLAY_FPS = 15  # Target FPS for display (lower = less CPU usage)
ENABLE_THREADING = True  # Use threading for face detection to improve responsiveness

# Layout settings (new)
LAYOUT_THEME = 'dark'  # UI theme (dark or light)

# Layout colors (new)
LAYOUT_COLORS = {
    'dark': {
        'background': '#333333',
        'panel_bg': '#222222',
        'panel_border': '#444444',
        'text': '#CCCCCC',
        'title': '#FFFFFF',
        'highlight': '#3498db',
        'recording': '#e74c3c'
    },
    'light': {
        'background': '#F0F0F0',
        'panel_bg': '#FFFFFF',
        'panel_border': '#CCCCCC',
        'text': '#333333',
        'title': '#000000',
        'highlight': '#2980b9',
        'recording': '#c0392b'
    }
}

# Panel titles (new)
PANEL_TITLES = {
    'camera': 'LIVE FEED OF THE CAMERA',
    'face1': 'FACE1 tracking',
    'face2': 'FACE2 tracking',
    'face3': 'FACE3 tracking',
    'face4': 'FACE4 tracking',
    'log': 'LOGGING MESSAGES',
    'empty': 'EMPTY PANEL'
}

# Log settings (new)
LOG_MAX_LINES = 1000  # Maximum number of lines to keep in the log

# Dynamic configuration settings

# Directory for configuration presets
CONFIG_PRESETS_DIR = "config_presets"
os.makedirs(CONFIG_PRESETS_DIR, exist_ok=True)

# Store default values
DEFAULT_CONFIG = {
    'CAMERA_DEVICE': CAMERA_DEVICE,
    'CAMERA_WIDTH': CAMERA_WIDTH,
    'CAMERA_HEIGHT': CAMERA_HEIGHT,
    'EYE_CONTACT_THRESHOLD': EYE_CONTACT_THRESHOLD,
    'VIDEO_CAPTURE_ENABLED': VIDEO_CAPTURE_ENABLED,
    'IMAGE_CAPTURE_ENABLED': IMAGE_CAPTURE_ENABLED,
    'DEBOUNCE_TIME': DEBOUNCE_TIME,
    'SCREENSHOT_DEBOUNCE_TIME': SCREENSHOT_DEBOUNCE_TIME,
    'VIDEO_DURATION': VIDEO_DURATION,
    'POST_GAZE_RECORD_TIME': POST_GAZE_RECORD_TIME,
    'FACE_REDETECTION_TIMEOUT': FACE_REDETECTION_TIMEOUT,
    'FACE_MARGIN_PERCENT': FACE_MARGIN_PERCENT,
    'HIGH_RES_ENABLED': HIGH_RES_ENABLED,
    'VIDEO_FPS': VIDEO_FPS,
    'PROCESSING_WIDTH': PROCESSING_WIDTH,
    'PROCESSING_HEIGHT': PROCESSING_HEIGHT,
    'FRAME_PROCESSING_INTERVAL': FRAME_PROCESSING_INTERVAL,
    'FACE_DETECTION_CONFIDENCE': FACE_DETECTION_CONFIDENCE,
    'FACE_DETECTION_MODEL': FACE_DETECTION_MODEL,
    'ENABLE_STREAMING': ENABLE_STREAMING,
    'STREAMING_PORT': STREAMING_PORT,
    'DISABLE_LOCAL_PREVIEW': DISABLE_LOCAL_PREVIEW,
    'STREAM_QUALITY': STREAM_QUALITY,
    'FACE_PERSISTENCE_FRAMES': FACE_PERSISTENCE_FRAMES,
    'USE_TKINTER_LAYOUT': USE_TKINTER_LAYOUT,
    'ENABLE_FULLSCREEN': ENABLE_FULLSCREEN,
    'LAYOUT_THEME': LAYOUT_THEME
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

def get_layout_color(color_name):
    """Get a color from the current theme."""
    theme = LAYOUT_THEME
    if theme not in LAYOUT_COLORS:
        theme = 'dark'  # Default to dark theme if the specified theme doesn't exist

    if color_name in LAYOUT_COLORS[theme]:
        return LAYOUT_COLORS[theme][color_name]
    else:
        # Return a default color if the specified color doesn't exist
        return '#333333' if theme == 'dark' else '#FFFFFF'