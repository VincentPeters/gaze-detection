import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, render_template
import logging
import config
from stream_buffer import StreamBuffer

# Suppress Flask logging to prevent console spam
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class StreamingServer:
    """
    Flask server for MJPEG streaming of camera feed and individual face feeds.
    """
    def __init__(self, stream_buffer, host='0.0.0.0', port=None):
        """
        Initialize the streaming server.

        Args:
            stream_buffer: StreamBuffer instance for accessing video frames
            host: Host to bind the server to (default: '0.0.0.0' - all interfaces)
            port: Port to bind the server to (default: None - use config.STREAMING_PORT)
        """
        self.stream_buffer = stream_buffer
        self.host = host
        self.port = port or config.STREAMING_PORT
        self.app = Flask(__name__)
        self.server_thread = None
        self.is_running = False

        # Set up routes
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/face_feed/<face_id>')(self.face_feed)
        self.app.route('/status')(self.status)

    def index(self):
        """Render the main page with all video streams."""
        stream_ids = self.stream_buffer.get_stream_ids()
        face_streams = [s for s in stream_ids if s.startswith('face_')]

        return render_template('index.html',
                              face_streams=face_streams,
                              has_main_stream='main' in stream_ids)

    def generate_frames(self, stream_id):
        """
        Generator function that yields MJPEG frames for a specific stream.

        Args:
            stream_id: ID of the stream to yield frames for
        """
        # Load or create the blank frame once
        import os
        if not os.path.exists('static/empty.jpg'):
            empty_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(empty_img, (0, 0), (320, 240), (0, 0, 0), -1)
            cv2.putText(empty_img, "No Stream Available", (40, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite('static/empty.jpg', empty_img)

        # Read the blank frame
        blank_image = cv2.imread('static/empty.jpg')
        if blank_image is None:
            # If file can't be read, create a simple blank image
            blank_image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank_image, "No Stream", (100, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Pre-encode the blank frame
        try:
            _, blank_jpeg = cv2.imencode('.jpg', blank_image,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_QUALITY])
            blank_frame_bytes = blank_jpeg.tobytes()
        except Exception as e:
            print(f"Error encoding blank frame: {e}")
            # Create a simple byte string if encoding fails
            blank_frame_bytes = b'Could not encode blank frame'

        while self.is_running:
            try:
                # Get the latest frame
                frame = self.stream_buffer.get_frame(stream_id)

                if frame is None or frame.size == 0:
                    # No frame available, yield the blank frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + blank_frame_bytes + b'\r\n')
                    time.sleep(1.0)  # Longer sleep when no frame is available
                    continue

                # Encode the frame as JPEG
                try:
                    result, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.STREAM_QUALITY])
                    if not result:
                        # If encoding failed, use blank frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + blank_frame_bytes + b'\r\n')
                        time.sleep(0.5)
                        continue

                    # Yield the frame in MJPEG format
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    # If encoding failed, use blank frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + blank_frame_bytes + b'\r\n')
                    time.sleep(0.5)
                    continue

                # Small sleep to control frame rate and CPU usage
                time.sleep(1.0 / 30)  # Limit to 30 FPS

            except Exception as e:
                print(f"Error in generate_frames: {e}")
                time.sleep(0.5)  # Sleep on error before retrying

    def video_feed(self):
        """Route for the main video feed."""
        return Response(self.generate_frames('main'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def face_feed(self, face_id):
        """Route for individual face feeds."""
        return Response(self.generate_frames(face_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def status(self):
        """Route for status information."""
        stream_ids = self.stream_buffer.get_stream_ids()
        return {
            'status': 'running',
            'streams': stream_ids,
            'timestamp': time.time()
        }

    def _run_server(self):
        """Run the Flask server in a separate thread."""
        self.app.run(host=self.host, port=self.port, debug=False, threaded=True, use_reloader=False)

    def start(self):
        """Start the streaming server in a separate thread."""
        if self.is_running:
            print("Streaming server is already running")
            return

        # Create necessary directories
        import os
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)

        # Create a default empty image if it doesn't exist
        if not os.path.exists('static/empty.jpg'):
            empty_img = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.rectangle(empty_img, (0, 0), (320, 240), (0, 0, 0), -1)
            cv2.putText(empty_img, "No Stream Available", (40, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite('static/empty.jpg', empty_img)

        self.is_running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"Streaming server started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the streaming server."""
        self.is_running = False
        # Note: Flask doesn't provide a clean way to stop the server
        # The daemon thread will be terminated when the main process exits
        print("Streaming server stopping (will terminate with main process)")