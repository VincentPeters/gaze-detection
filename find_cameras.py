#!/usr/bin/env python
"""
Combined script to find available cameras.
"""

import cv2

# Default configuration (if the src.utils.config module is not available)
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class Camera:
    """
    Camera class for handling camera operations.
    """
    def __init__(self, camera_index=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        """
        Initialize the camera.

        Args:
            camera_index: Index of the camera to use.
            width: Width of the camera frame.
            height: Height of the camera frame.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """
        Open the camera.

        Returns:
            bool: True if the camera was opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def read(self):
        """
        Read a frame from the camera.

        Returns:
            tuple: (success, frame) where success is True if the frame was read successfully.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """
        Release the camera.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """
        Context manager entry.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        """
        self.release()


def list_available_cameras(max_cameras=10):
    """
    List all available camera devices by trying to open each index.

    Args:
        max_cameras: Maximum number of cameras to check.

    Returns:
        list: List of available camera indices.
    """
    available_cameras = []

    print("Searching for available cameras...")

    # Try camera indices from 0 to max_cameras-1
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} is available")
                available_cameras.append(i)
            cap.release()
        else:
            print(f"Camera index {i} is not available")

    if not available_cameras:
        print("No cameras were found. Please check your camera connections.")
    else:
        print(f"Found {len(available_cameras)} available camera(s) at indices: {available_cameras}")

    return available_cameras


if __name__ == "__main__":
    list_available_cameras(max_cameras=10)
