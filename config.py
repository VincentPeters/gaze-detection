# Configuration file for the eye contact detection application

# Paths
MODEL_PATH = "models/model_weights.pkl"
VIDEOS_DIR = "eye_contact_videos"
SCREENSHOTS_DIR = "eye_contact_screenshots"
FACES_DIR = "faces"

# Eye contact detection settings
EYE_CONTACT_THRESHOLD = 0.3  # Lowered from 0.5 to be less strict
FACE_MARGIN_PERCENT = 60  # Percentage of margin to add around detected faces

# Timing settings
DEBOUNCE_TIME = 5.0  # Seconds between video recordings for the same face
SCREENSHOT_DEBOUNCE_TIME = 1.0  # Seconds between screenshots for the same face
VIDEO_DURATION = 6.0  # Duration of recorded videos in seconds
POST_GAZE_RECORD_TIME = 1.0  # Continue recording for this many seconds after eye contact is lost
FACE_REDETECTION_TIMEOUT = 1.0  # How long to keep tracking a face after detection fails

# Video recording settings
VIDEO_FPS = 20  # Frames per second for recorded videos

# Resolution settings
HIGH_RES_ENABLED = True  # Enable high-resolution capture
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

# Face detection settings
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection
FACE_DETECTION_MODEL = 1  # MediaPipe model selection (0 for close-range, 1 for full-range)

# Window settings
MAIN_WINDOW_NAME = 'Face Detection'
FACE_WINDOW_WIDTH = 200  # Width of individual face windows
FACE_WINDOW_HEIGHT = 200  # Height of individual face windows
MAIN_WINDOW_POSITION = (50, 50)  # (x, y) position of the main window