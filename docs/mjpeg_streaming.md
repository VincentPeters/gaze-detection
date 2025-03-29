# MJPEG Streaming Implementation Plan

## Overview

This document outlines the plan to add browser-based video streaming to the face detection application using MJPEG over HTTP with Flask. The implementation will enable users to view the main camera feed and individual face feeds in a web browser.

## Requirements

- Stream main camera feed and individual face feeds to a browser
- Target resolution: 1080p where possible
- Local access only (no authentication required)
- Option to disable local preview windows while keeping streaming active
- Minimal changes to existing codebase

## Implementation Plan

### 1. Frame Sharing Mechanism ✅

**Files to modify:**
- `main.py` - Add frame buffer for sharing frames with streaming component ✅

**New files:**
- `stream_buffer.py` - Thread-safe buffer to hold frames for streaming ✅

**Tasks:**
- Implement a thread-safe frame buffer class ✅
- Modify `process_frame()` method to store frames in the buffer ✅
- Ensure buffer updates don't impact performance ✅

### 2. Flask Server Implementation ✅

**New files:**
- `streaming_server.py` - Flask application with streaming endpoints ✅
- `templates/index.html` - Web UI for displaying streams ✅

**Tasks:**
- Create Flask application with routes for each stream ✅
- Implement MJPEG encoding and streaming functionality ✅
- Create HTML template for the web interface ✅
- Add static CSS for layout and styling ✅

### 3. Integration and Configuration ✅

**Files to modify:**
- `config.py` - Add streaming configuration options ✅
- `main.py` - Initialize and start streaming server ✅

**Tasks:**
- Add configuration options for streaming (port, enable/disable) ✅
- Add option to disable local preview windows ✅
- Integrate Flask server startup into the main application ✅
- Ensure clean shutdown of streaming server ✅

### 4. Web UI Implementation ✅

**New files:**
- `templates/index.html` - Main page template ✅
- `static/css/style.css` - Styling for the web interface ✅

**Tasks:**
- Create responsive layout for multiple video streams ✅
- Add basic styling and layout ✅
- Display stream status and application info ✅

## Technical Details

### Frame Buffer Architecture
- Thread-safe queue or circular buffer for each stream
- Configurable max buffer size to manage memory usage
- Independent buffers for main feed and face feeds

### Streaming Server
- Run in a dedicated thread within the main process
- Flask routes for:
  - `/` - Main web interface
  - `/video_feed` - Main camera MJPEG stream
  - `/face_feed/<face_id>` - Individual face MJPEG streams
  - `/status` - Simple status endpoint

### Configuration Options
```
# Streaming configuration
ENABLE_STREAMING = True  # Enable/disable streaming server
STREAMING_PORT = 5000    # Port for the streaming server
DISABLE_LOCAL_PREVIEW = False  # Disable local preview windows but keep streaming
STREAM_QUALITY = 90      # JPEG quality for streams (0-100)
```

## Implementation Sequence

1. Implement frame buffer mechanism ✅
2. Create basic Flask server with main feed streaming ✅
3. Add face feed streaming ✅
4. Implement configuration options ✅
5. Create web UI ✅
6. Test and optimize performance

## Future Improvements

- WebRTC for lower latency (potential future upgrade)
- WebSocket for bidirectional communication
- Stream quality control in the web interface
- Recording controls in the web interface