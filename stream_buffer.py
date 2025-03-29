import threading
import numpy as np
import time

class StreamBuffer:
    """
    Thread-safe buffer for storing frames to be streamed via MJPEG.
    """
    def __init__(self, buffer_size=1):
        """
        Initialize a buffer for a single stream.

        Args:
            buffer_size: Number of frames to keep in the buffer (default: 1)
        """
        self.buffer = {}
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.last_update_time = {}

    def update_frame(self, stream_id, frame):
        """
        Update the frame for a specific stream.

        Args:
            stream_id: Identifier for the stream (e.g., 'main' or 'face_1')
            frame: The new frame (numpy array)
        """
        if frame is None:
            return

        # Make a copy of the frame to avoid reference issues
        frame_copy = frame.copy() if frame is not None else None

        with self.lock:
            self.buffer[stream_id] = frame_copy
            self.last_update_time[stream_id] = time.time()

    def get_frame(self, stream_id):
        """
        Get the latest frame for a specific stream.

        Args:
            stream_id: Identifier for the stream

        Returns:
            The latest frame or None if no frame is available
        """
        with self.lock:
            return self.buffer.get(stream_id, None)

    def get_stream_ids(self):
        """
        Get all active stream IDs.

        Returns:
            List of stream IDs
        """
        with self.lock:
            return list(self.buffer.keys())

    def get_last_update_time(self, stream_id):
        """
        Get the timestamp of the last update for a stream.

        Args:
            stream_id: Identifier for the stream

        Returns:
            Timestamp of last update or 0 if stream doesn't exist
        """
        with self.lock:
            return self.last_update_time.get(stream_id, 0)

    def remove_stream(self, stream_id):
        """
        Remove a stream from the buffer.

        Args:
            stream_id: Identifier for the stream to remove
        """
        with self.lock:
            if stream_id in self.buffer:
                del self.buffer[stream_id]
            if stream_id in self.last_update_time:
                del self.last_update_time[stream_id]