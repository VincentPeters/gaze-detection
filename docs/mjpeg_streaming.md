# MJPEG Streaming Implementation

This document outlines the implementation of MJPEG streaming in the face detection application.

## Overview

The application now operates exclusively through web streaming, with no local OpenCV or Tkinter windows. This provides better performance and enables remote viewing of the face detection system.

## Configuration Options

In `config.py`, the following options control the streaming behavior:

- `ENABLE_STREAMING = True`: Enables the MJPEG streaming server
- `STREAMING_PORT = 8080`: The port on which the streaming server runs (default: 8080)
- `DISABLE_LOCAL_PREVIEW = True`: Local preview windows are disabled by default
- `STREAM_QUALITY = 90`: JPEG quality for streams (0-100)

## Implementation Details

### Stream Buffer

The `stream_buffer.py` file implements a thread-safe buffer for frames:

- Uses a dictionary of queues to store frames for each stream
- Each stream (main camera and individual faces) has its own buffer
- Thread-safe operations for updating and retrieving frames

### Streaming Server

The `streaming_server.py` file implements a Flask server for MJPEG streaming:

- Runs in a separate thread to avoid blocking the main application
- Provides routes for the main web interface and individual video feeds
- Serves a responsive web UI that displays all streams

### Visual Indicators

The streamed video includes enhanced visual indicators:

1. **Main Feed Indicators**:
   - Bounding boxes around detected faces
   - Eye contact status text

2. **Face Feed Indicators**:
   - Color-coded borders (green for eye contact, red otherwise)
   - Bold text showing eye contact score
   - Face ID label
   - "REC" indicator when recording

## Running the Application

To run the application with streaming:

1. Start the application: `python main.py`
2. Open a web browser and navigate to `http://localhost:8080`

## Performance Considerations

- The application may perform better now that it's not rendering local windows
- Streaming quality can be adjusted in the config file to reduce CPU usage
- Multiple clients can connect to the streaming server simultaneously

## Troubleshooting

- If you see an "Address already in use" error, change the `STREAMING_PORT` value in `config.py`
- Make sure your firewall allows connections on the configured port if accessing remotely

## Future Improvements

- Add authentication for the web interface
- Implement WebRTC for lower latency streaming
- Add stream selection options in the web UI
- Implement recording controls in the web interface