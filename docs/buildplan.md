Here's a step-by-step development plan you could follow to create the real-time gaze detection and face capture application. The steps are arranged in a way that you can break them down and work on them piece by piece, potentially using AI assistance at each step.

---

## STEP 1: SETUP YOUR DEVELOPMENT ENVIRONMENT ✅

1. **Select Your Computing Platform** ✅
   - Choose between Raspberry Pi, Jetson Nano, or a mini PC.
   - Install the operating system (e.g., Raspberry Pi OS, Ubuntu, or any Linux distro).

2. **Install Required Dependencies** ✅
   - Update and upgrade the system (`sudo apt-get update && sudo apt-get upgrade`).
   - Install Python 3.x (if not already available).
   - Install Python packages:
     ```bash
     pip install opencv-python mediapipe numpy
     ```
     > *Optionally:* If you plan to use a deep learning model (OpenCV DNN, Dlib, etc.), install those dependencies as well (`pip install dlib`, `pip install torch`, etc.).

3. **Camera Configuration** ✅
   - If using a USB webcam or DSLR, ensure your OS recognizes it (`ls /dev/video*` on Linux).
   - If using a Raspberry Pi Camera module, enable the camera interface (`sudo raspi-config` -> Interfaces -> Camera).
   - Test that you can capture a basic image (e.g., using `libcamera` on Pi or `ffmpeg`/`v4l2` on Linux).

---

## STEP 2: CAPTURE AND DISPLAY A LIVE VIDEO FEED ✅

1. **Write a Basic Script to Capture Frames** ✅
   ```python
   import cv2

   def main():
       cap = cv2.VideoCapture(0)  # Adjust index if multiple cams
       while True:
           ret, frame = cap.read()
           if not ret:
               break

           cv2.imshow('Live Feed', frame)

           # Exit on 'q'
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

       cap.release()
       cv2.destroyAllWindows()

   if __name__ == '__main__':
       main()
   ```
   - Ensure the camera feed appears in a window labeled "Live Feed."
   - Tweak resolution if necessary (e.g., `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)`).

2. **Test on One Screen** ✅
   - For now, just display on your primary screen.
   - Make sure the video feed is stable and at an acceptable frame rate.

---

## STEP 3: BASIC FACE DETECTION ✅

1. **Implement Face Detection (Using MediaPipe or OpenCV)** ✅
   - For simplicity, start with MediaPipe's face detection. Install it (`pip install mediapipe`).
   - Example snippet:
     ```python
     import mediapipe as mp
     import cv2

     mp_face_detection = mp.solutions.face_detection
     mp_drawing = mp.solutions.drawing_utils

     face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
     ```

2. **Integrate Face Detection Into the Live Feed** ✅
   ```python
   import cv2
   import mediapipe as mp

   mp_face_detection = mp.solutions.face_detection
   mp_drawing = mp.solutions.drawing_utils

   def main():
       cap = cv2.VideoCapture(0)
       with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
           while True:
               ret, frame = cap.read()
               if not ret:
                   break

               # Convert to RGB for MediaPipe
               frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               results = face_detector.process(frame_rgb)

               # Draw face detection annotations
               if results.detections:
                   for detection in results.detections:
                       mp_drawing.draw_detection(frame, detection)

               cv2.imshow('Live Feed with Face Detection', frame)
               if cv2.waitKey(1) & 0xFF == ord('q'):
                   break

       cap.release()
       cv2.destroyAllWindows()

   if __name__ == '__main__':
       main()
   ```
   - Verify that bounding boxes appear around faces.

3. **Tune Face Detection Parameters** ✅
   - Adjust `model_selection` (0 or 1) depending on distance. For your use case (1-6 meters), keep `model_selection=1` which is for long-range.
   - Adjust `min_detection_confidence` if you get too many false positives or missed detections.

---

## STEP 4: FACIAL LANDMARKS AND GAZE DETECTION ✅

1. **Switch to MediaPipe Face Mesh (Optional)** ✅
   - MediaPipe Face Mesh provides detailed landmarks, including iris.
   - Alternatively, you can keep using Face Detection if you're only looking for bounding boxes. But for gaze detection, Face Mesh is more powerful.

2. **Extract Key Points (Eyes, Iris) for Gaze** ✅
   - If using Face Mesh:
     ```python
     mp_face_mesh = mp.solutions.face_mesh
     with mp_face_mesh.FaceMesh(
         max_num_faces=10,
         refine_landmarks=True,       # This enables iris landmarks
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5) as face_mesh:
         ...
     ```
   - The `refine_landmarks=True` argument includes iris keypoints.
   - Track these landmarks to see if the person's gaze is aligned with the camera.

3. **Compute "Looking Straight at Camera"** ✅
   - For a simple heuristic, measure if the head is frontal and the irises are centered.
   - Alternatively, do a head pose estimation using known 3D face points (`solvePnP` approach with OpenCV).
   - Keep it basic at first: check if the eyes are not drastically looking left/right. If using iris landmarks, you can see if `iris_center` is in the middle region of the eye.
   - Write a function `is_looking_at_camera(landmarks)` that returns True if the face is directed at the camera.

4. **Annotate Gaze** ✅
   - Draw a green bounding box or put text "Looking" if `is_looking_at_camera == True`.
   - Draw another color or no label if not looking.

---

## STEP 5: CAPTURE FACE IMAGE WHEN LOOKING ✅

1. **Crop and Save the Face** ✅
   - When `is_looking_at_camera()` returns True:
     - Convert the face bounding box or landmarks to pixel coordinates.
     - Crop that region from the frame.
     - Save the cropped image (`cv2.imwrite('captures/face_xxx.jpg', cropped_img)`).
   - Use a timestamp for file naming. E.g. `face_{time.time()}.jpg`.

2. **Avoid Duplicates / Spamming** ✅
   - Implement a cooldown. For each face detection event, wait 1-2 seconds before capturing again.
   - You can store the last capture timestamp. If a new detection occurs <2 seconds after the last, skip.
   - Or track each face with an ID (using a simple tracker). If the same face ID is still in frame, don't re-capture immediately.

3. **Verify Results** ✅
   - Test to see if the system captures images only when you look at the camera.
   - Adjust thresholds if it triggers too often or misses gazes.

---

## STEP 6: DUAL-SCREEN DISPLAY ✅

1. **Preparing for Two Displays** ✅
   - Confirm that the OS recognizes two monitors. In Raspberry Pi OS, for instance, go to Display Settings and set them up in "Extended Desktop" mode.
   - In a minimal window manager, you might only have a single desktop spanning two monitors. The main idea is to position different windows on each screen.

2. **Window Placement** ✅
   - In OpenCV, you can do something like:
     ```python
     cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
     cv2.moveWindow('Live Feed', x=0, y=0)  # Suppose this is the left screen
     ```
     Then for the slideshow:
     ```python
     cv2.namedWindow('Slideshow', cv2.WINDOW_NORMAL)
     cv2.moveWindow('Slideshow', x=1920, y=0)  # Move to the right screen if each screen is 1920 wide
     ```
   - Alternatively, use a different approach for the slideshow (like a Tkinter or PyQt app) placed on the second display.

3. **Implement a Basic Slideshow** ✅
   - Use a separate Python thread or script to read from the "captures/" folder and display images in full screen.
   - Simple approach in OpenCV (pseudo-code):
     ```python
     import glob
     import time

     while True:
         images = glob.glob('captures/*.jpg')
         for img_path in images:
             img = cv2.imread(img_path)
             cv2.imshow('Slideshow', img)
             key = cv2.waitKey(2000)  # Display each for 2 seconds
             if key == 27:  # Esc to exit
                 break
         # maybe shuffle or keep order
     ```
   - Alternatively, you can create a fullscreen Tkinter/PyQt app that updates an `Label` widget with each image in turn.

---

## STEP 7: STORAGE AND CONNECTIVITY

1. **Local Storage**
   - Confirm that the capture folder is large enough.
   - Possibly implement a rotating scheme to delete older files if you need to save space.

2. **Optional Cloud Backup**
   - If you want to store images online, integrate something like AWS S3 or Google Drive.
   - For example, each time you save a face, call an upload function (or do it asynchronously in another thread).

3. **Remote Monitoring**
   - Optionally, create a small Flask server that shows the latest captures or the real-time feed.
   - Or enable SSH/VNC for direct remote control.

---

## STEP 8: TEST, TWEAK, AND OPTIMIZE

1. **Performance Tuning**
   - If FPS is low, reduce the resolution or switch to a faster model.
   - Use multi-threading (frame capture in one thread, face detection in another).
   - Consider hardware accelerators (Intel NCS, Google Coral, Jetson GPU) if needed.

2. **Nighttime Testing**
   - Check if the camera sees well at night.
   - If not, add IR lighting or choose a camera with better low-light performance.
   - Adjust detection confidence for low-light conditions if necessary.

3. **Refine Gaze Thresholds**
   - Tweak angles or iris center boundaries for what counts as "looking."
   - Ensure you don't get frequent false positives from partial glances.

4. **User Experience & Reliability**
   - Ensure the system is stable over long periods.
   - Add error handling if the camera disconnects. Possibly auto-restart camera capture.

---

## STEP 9: DEPLOYMENT AND MAINTENANCE

1. **Autorun on Startup**
   - If using Linux, create a systemd service or `cron @reboot` job that launches your main script.
   - Ensure the second script for slideshow also starts automatically.

2. **Watchdog**
   - Consider a watchdog that restarts your script if it crashes.

3. **Ongoing Maintenance**
   - Clean or align the camera lens periodically.
   - Manage disk space in `captures/`.
   - Update software libraries for security patches.

4. **Expand / Add Features**
   - If needed, you can add face recognition (who is looking?), additional logging, or integration with web frameworks for real-time dashboards.

---

### Conclusion

By following these steps, you'll develop a robust application that displays a live feed of the camera, detects faces, determines who's looking directly at the camera, captures cropped face images, and displays them on a second screen slideshow. Each step can be tackled incrementally:

1. **Environment Setup** ✅
2. **Live Feed** ✅
3. **Face Detection** ✅
4. **Gaze Detection** ✅
5. **Face Capture Logic** ✅
6. **Dual-Screen Integration** ✅
7. **Storage/Connectivity**
8. **Testing & Optimization**
9. **Deployment & Maintenance**

Good luck with your project!