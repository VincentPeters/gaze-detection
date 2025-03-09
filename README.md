# Eye Contact Detection and Recording

This application uses computer vision and machine learning to detect when people are making eye contact with the camera. It automatically captures screenshots and records video clips when eye contact is detected, making it useful for engagement analysis, research, and interactive applications.

## Features

- **Real-time eye contact detection** using a pre-trained AI model
- **High-resolution screenshot capture** when eye contact is detected
- **Automatic video recording** of faces making eye contact
- **Multi-face tracking** with individual windows for each detected face
- **Robust face tracking** that continues even when detection temporarily fails
- **Configurable parameters** through a central configuration file

## Requirements

- Python 3.8 or higher
- Webcam or camera device
- Dependencies listed in `requirements.txt`

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gaze-detection.git
cd gaze-detection
```

### 2. Create a Virtual Environment

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Using the Run Scripts

For convenience, you can use the provided run scripts:

- **Windows**: Double-click `run.bat` or run it from the command line
- **macOS/Linux**: Make the script executable with `chmod +x run.sh` and then run `./run.sh`

These scripts will automatically activate the virtual environment, run the application, and deactivate the environment when done.

### Manual Execution

Alternatively, you can run the application manually:

```bash
# Activate the virtual environment first
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Then run the application
python main.py
```

The application will:
1. Open your webcam
2. Detect faces in the video stream
3. Analyze each face for eye contact
4. Take screenshots and record videos when eye contact is detected
5. Display the results in real-time

Press 'q' to quit the application.

## Configuration

You can customize the application's behavior by editing the `config.py` file:

### Eye Contact Detection

- `EYE_CONTACT_THRESHOLD`: Probability threshold for eye contact detection (default: 0.3)
- `FACE_MARGIN_PERCENT`: Percentage of margin to add around detected faces (default: 60%)

### Media Capture

- `SCREENSHOT_DEBOUNCE_TIME`: Seconds between screenshots for the same face (default: 1.0)
- `DEBOUNCE_TIME`: Seconds between video recordings for the same face (default: 5.0)
- `VIDEO_DURATION`: Duration of recorded videos in seconds (default: 6.0)
- `VIDEO_FPS`: Frames per second for recorded videos (default: 20)

### Resolution Settings

- `HIGH_RES_ENABLED`: Enable high-resolution capture (default: True)
- `DISPLAY_WIDTH`: Width for display and processing (default: 640)
- `DISPLAY_HEIGHT`: Height for display and processing (default: 480)
- `CAMERA_RESOLUTIONS`: List of resolutions to try (in order of preference)

### Output Directories

- `VIDEOS_DIR`: Directory for saving video recordings
- `SCREENSHOTS_DIR`: Directory for saving screenshots
- `FACES_DIR`: Directory for temporary face images

## Understanding the Output

- **Green box**: Face is making eye contact
- **Red box**: Face is not making eye contact
- **Orange dashed box**: Face is being tracked but detection temporarily failed
- **Red circle** in face window: Currently recording video

## Troubleshooting

### Camera Not Detected

If the camera is not detected, check:
1. That your webcam is properly connected
2. That no other application is using the camera
3. That you have the necessary permissions to access the camera

### Performance Issues

If the application is running slowly:
1. Lower the camera resolution in `config.py`
2. Disable high-resolution capture by setting `HIGH_RES_ENABLED = False`
3. Increase the eye contact threshold to reduce false positives

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses MediaPipe for face detection
- Eye contact detection model based on https://github.com/rehg-lab/eye-contact-cnn