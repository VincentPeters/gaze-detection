Below is a **step-by-step development plan** showing how you can **integrate a pre-trained AI model** (designed for direct eye contact detection) into your **existing face detection and cropping workflow**. This assumes you have:

- Your current code base (from the `main.py` example).
- A **pre-trained model** that, given a cropped face image, outputs a probability of "direct eye contact."

The following plan breaks down the process into clear, incremental tasks.

---

## 1. Acquire the Pre-Trained Model ✅

1. **Download/Obtain the Model Weights** ✅
   - Depending on the model's framework (e.g., PyTorch, TensorFlow), you'll receive either a `.pt`/`.pth` file (PyTorch) or a `.h5`/`.pb` file (TensorFlow).
   - Store this file locally in your project (e.g., `models/eye_contact_detector.pth`).

2. **Download/Obtain Model Code or Architecture** ✅
   - If using PyTorch, you'll need a `.py` file or code snippet that defines the model class/architecture and can load the weights.
   - If using TensorFlow, you might do something like `tf.keras.models.load_model('path/to/model')`.

3. **Install Dependencies** ✅
   - If it's a PyTorch model:
     ```bash
     pip install torch torchvision
     ```
   - If it's TensorFlow:
     ```bash
     pip install tensorflow
     ```
   - The exact dependencies vary based on the model. Check the model's documentation.

---

## 2. Integrate Model Loading in Your Python Code ✅

1. **Create a New Python Module for Inference** ✅
   - For clarity, place your model loading and inference logic in, for example, `eye_contact_model.py`.
   - Pseudocode for a PyTorch example:

     ```python
     # eye_contact_model.py
     import torch
     import torch.nn as nn
     import torchvision.transforms as T
     from PIL import Image

     class EyeContactModel(nn.Module):
         def __init__(self):
             super().__init__()
             # define layers or load an existing architecture
             # e.g. self.net = <some pretrained backbone, e.g. ResNet>

         def forward(self, x):
             # forward pass
             return self.net(x)

     # Inference wrapper
     class EyeContactDetector:
         def __init__(self, model_path):
             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
             self.model = EyeContactModel()
             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
             self.model.eval()
             self.model.to(self.device)

             # define transforms
             self.transform = T.Compose([
                 T.Resize((224, 224)),   # or whatever input size your model expects
                 T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
             ])

         def predict_eye_contact_probability(self, image_bgr):
             # Convert BGR (OpenCV) -> RGB (PIL)
             image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
             pil_img = Image.fromarray(image_rgb)

             # apply transforms
             input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

             # forward pass
             with torch.no_grad():
                 output = self.model(input_tensor)
                 # assume output is a single sigmoid logit or [prob_no_contact, prob_contact]
                 if output.shape[-1] == 1:
                     # single scalar -> pass through sigmoid
                     prob_contact = torch.sigmoid(output).item()
                 else:
                     # assume two-class softmax
                     prob_contact = torch.softmax(output, dim=1)[0, 1].item()

             return prob_contact
     ```
   - The details depend on your exact model. The key is to **accept a cropped BGR face image** from OpenCV, transform it, pass it to the network, and output a probability score.

2. **Load the Model in `main.py`** ✅
   - Inside `main.py` (or whichever file drives your main loop), **import** this new module.
   - Initialize the detector once (e.g., `detector = EyeContactDetector('models/eye_contact_detector.pth')`).

---

## 3. Modify Your Face Cropping Logic to Call the Model ✅

1. **Locate Where You Crop the Face** ✅
   - In your current `main.py`, you already extract a `face_img`:

     ```python
     face_img = frame[y:y+h, x:x+w].copy()
     ```
   - Right after cropping (and possibly resizing for display), call your new model's inference method, e.g.:

     ```python
     prob_eye_contact = detector.predict_eye_contact_probability(face_img)
     ```

2. **Threshold & Display** ✅
   - Suppose you define a threshold of `0.5` to decide "direct eye contact":
     ```python
     if prob_eye_contact > 0.5:
         # The model says this face is likely making eye contact
         # Mark it somehow
         label = f"Eye Contact: {prob_eye_contact:.2f}"
     else:
         label = f"No Eye Contact: {prob_eye_contact:.2f}"
     ```
   - You can then draw the label on the face window or main window:
     ```python
     cv2.putText(face_img, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
     ```
   - If you want to highlight it in the main frame, you could place text near the bounding box:
     ```python
     cv2.putText(display_frame, label, (x, y - 25),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     ```

3. **Consider Performance**
   - In real time, you might be running multiple faces each frame. Keep in mind that each face triggers a model inference. If performance is too low, you have options:
     - **Skip frames** (e.g., run eye contact detection every 3rd or 5th frame).
     - **Use a smaller input size** or a more optimized model.
     - **Batch inference** (if your model supports it) by collecting all face crops in one pass.
     - If it's still slow, consider GPU usage or a faster CPU.

---

## 4. Optional: Fine-Tune or Adjust the Model ✅

1. **Check Real-World Accuracy** ✅
   - Test the system with different users, lighting conditions, face distances, and angles.
   - If the results are not consistent, you might need to gather a few examples (images) from your specific environment, label them (eye contact vs. no eye contact), and **fine-tune** the model.

2. **Calibration** ✅
   - If the model is slightly off in practice (e.g., it's giving 40% when the user is truly looking), you can adjust the threshold or calibrate the model with a small local dataset. For instance, you might find that `prob_eye_contact > 0.3` is the best cutoff for your environment.
   - Lowered the threshold from 0.5 to 0.3 to make eye contact detection less strict and work at greater distances.

3. **Multi-Face Handling**
   - Because you detect multiple faces, confirm that the bounding boxes/crops are large enough (≥80×80 pixels, ideally 224×224 for many CNNs). If the bounding box is too small (distant faces), the model might fail.
   - You might need to restrict classification to faces of a certain minimum bounding box size to avoid false negatives on tiny faces.

---

## 5. Testing & Debugging

1. **Frame-Rate and Latency Checks**
   - Print out or measure how many frames per second (FPS) you get.
   - If FPS is too low (e.g., below 10–15), try reducing resolution or skipping frames for inference.

2. **Edge Cases**
   - People wearing glasses (reflections can trick gaze models).
   - Faces at extreme angles (though we only want direct eye contact, so that might not matter if face is turned sideways).
   - Low light conditions. Check if the model can still detect contact in near-darkness.

3. **Visual Indicators**
   - Make sure you have clear, real-time feedback. For example:
     - Green text/box if the model sees direct gaze.
     - Red text/box if not.
   - This helps you or your users quickly verify correctness.

---

## 6. Deployment & Maintenance

1. **Packaging** ✅
   - If you plan to distribute or run this on multiple machines, consider packaging your dependencies. Use `requirements.txt` or a virtual environment.
   - Keep the model file (`.pth`, `.pb`, etc.) in a known location or embed it in your distribution.
   - Specify exact versions of critical dependencies (OpenCV 4.11.0.86, MediaPipe 0.10.21) to ensure compatibility.

2. **Auto-Run / Startup**
   - If you want your system to start on boot (for a kiosk-like setup), create a script or systemd service that launches your Python script automatically.

3. **Model Updates**
   - If you later find a better or more accurate pre-trained model, you can swap it in by updating `eye_contact_model.py` (as long as the input/output interfaces remain the same).
   - For major changes (e.g., different input sizes, more classes), adapt your code accordingly (resizing, different thresholding, etc.).

4. **Future Enhancements**
   - Integrate a short "temporal smoothing" to reduce flickering predictions (e.g., keep a rolling average of the last N inferences for each face).
   - Track each face over time (use an ID or a tracker) so if a person consistently has "eye contact," you can trigger additional logic (like capturing a snapshot, playing a sound, etc.).

---

## 7. Additional Features ✅

1. **Media Capture for Eye Contact** ✅
   - Automatically capture both screenshots and video clips when direct eye contact is detected.
   - Implement separate debounce mechanisms for screenshots (1 second) and videos (5 seconds) to control capture frequency.
   - Use multithreading to handle video recording without affecting the main application performance.
   - Save media with timestamps and face identifiers for later analysis.
   - This feature is useful for:
     - Collecting training data for model improvement
     - Creating a record of engagement for analysis
     - Building attention-based interactive applications
     - Studying user behavior and attention patterns

2. **Enhanced Video Recording** ✅
   - Save videos in MP4 format for better compatibility and compression.
   - Expand the face bounding box to capture more context around the face.
   - Continue recording for a short time after eye contact is lost to create smoother videos.
   - Use a queue-based approach to ensure all frames are captured properly.

3. **Robust Face Tracking** ✅
   - Continue tracking faces even when detection temporarily fails.
   - Store the last known position of each face and continue recording from that position.
   - Implement a timeout mechanism to stop tracking faces that haven't been detected for too long.
   - Provide visual feedback when using the last known position for tracking.

4. **Configuration System** ✅
   - Create a centralized configuration file (`config.py`) to store all adjustable parameters.
   - Make it easy to tune the system by changing values in a single location.
   - Include settings for:
     - Eye contact threshold (lowered to 0.3 for better detection at distance)
     - Camera resolution options
     - Video and screenshot settings
     - Face detection parameters
     - UI and display settings
   - Display current configuration values in the UI for easy reference.

---

## Summary of the Integration Steps

1. **Load the Model** ✅
   - Create or adapt a module (e.g., `eye_contact_model.py`) that loads the pre-trained weights.

2. **In `main.py`, Initialize the Model** ✅
   - `detector = EyeContactDetector('path/to/weights.pth')` (or your framework's equivalent).

3. **For Each Face** ✅
   - Crop the face from your existing bounding box logic.
   - Pass the cropped face to the model method: `prob_eye_contact = detector.predict_eye_contact_probability(face_img)`.
   - Use a threshold to decide if it's "direct eye contact."
   - Overlay results (probability, label, color-coded bounding box) on the displayed frames.

4. **Test, Tweak, and Optimize** ✅
   - Evaluate performance (FPS).
   - Adjust threshold or frame skipping if needed.
   - Lower the eye contact threshold to 0.3 for better detection at greater distances.
   - Create a configuration file for easy parameter tuning.

5. **Additional Features** ✅
   - Implement both screenshot and video recording for faces making direct eye contact.
   - Add separate debounce mechanisms to control screenshot and video recording frequency.
   - Use multithreading to handle recording without affecting main application performance.
   - Save videos in MP4 format with expanded face regions.
   - Continue recording briefly after eye contact is lost for smoother videos.
   - Implement robust face tracking to handle temporary detection failures.
   - Create a centralized configuration system for easy parameter adjustment.

By following these steps, you'll integrate a **pre-trained direct eye contact detector** into your existing face detection pipeline, obtaining a reliable measure of whether someone is looking directly at the camera even when pupil-based methods fail for smaller or distant faces.