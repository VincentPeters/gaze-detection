import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import re

class LayoutManager:
    """
    Manages the Tkinter-based layout for the application.
    Provides a multi-panel interface with support for camera feed,
    face tracking panels, logging, and an empty panel for future features.
    """

    def __init__(self, root=None, enable_fullscreen=False):
        """
        Initialize the layout manager.

        Args:
            root: Optional Tkinter root window. If None, a new one will be created.
            enable_fullscreen: Whether to start in fullscreen mode.
        """
        # Create or use provided root window
        if root is None:
            self.root = tk.Tk()
            self.owns_root = True
        else:
            self.root = root
            self.owns_root = False

        # Set window title and configure
        self.root.title("Face Tracking with Eye Contact Detection")
        self.root.configure(background='#000000')

        # Track if we're in fullscreen mode
        self.is_fullscreen = enable_fullscreen
        if self.is_fullscreen:
            self.root.attributes('-fullscreen', True)

        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # Set initial window size (80% of screen if not fullscreen)
        if not self.is_fullscreen:
            width = int(self.screen_width * 0.8)
            height = int(self.screen_height * 0.8)
            self.root.geometry(f"{width}x{height}")

        # Define fixed panel sizes based on window size - adjusted for better proportions
        self.camera_width = int(width * 0.95) if 'width' in locals() else 950  # Full width
        self.camera_height = int(height * 0.5) if 'height' in locals() else 400  # Half height
        self.face_panel_width = int(width * 0.95) if 'width' in locals() else 950  # Full width
        self.face_panel_height = int(height * 0.3) if 'height' in locals() else 240  # 30% of height
        self.log_panel_height = int(height * 0.15) if 'height' in locals() else 120  # 15% of height

        # Prevent window from being resized smaller than minimum size
        min_width = self.camera_width + 20  # Padding
        min_height = self.camera_height + self.face_panel_height + self.log_panel_height + 20  # Padding
        self.root.minsize(min_width, min_height)

        # Configure grid layout to be responsive
        self.root.grid_columnconfigure(0, weight=1)  # Full width column
        self.root.grid_columnconfigure(1, weight=1)  # Full width column
        self.root.grid_rowconfigure(0, weight=5, minsize=self.camera_height)  # Camera feed row
        self.root.grid_rowconfigure(1, weight=3, minsize=self.face_panel_height)  # Face panels row
        self.root.grid_rowconfigure(2, weight=2, minsize=self.log_panel_height)  # Log row

        # Callback functions for keyboard shortcuts
        self.config_callback = None
        self.reset_config_callback = None

        # Create the main panels
        self._create_panels()

        # Initialize image placeholders for panels
        self.camera_image = None
        self.face_images = [None, None, None, None]

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Process initial events to ensure widgets are properly initialized
        self.root.update_idletasks()

        # Force an initial update to set panel sizes
        self.root.after(100, self._set_initial_panel_sizes)

    def _set_initial_panel_sizes(self):
        """Force panels to have consistent initial sizes."""
        # Set fixed sizes for camera panel
        self.camera_label.config(width=self.camera_width, height=self.camera_height)

        # Set fixed sizes for face panels
        face_width = self.face_panel_width // 2 - 4
        face_height = self.face_panel_height // 2 - 4
        for label in self.face_labels:
            label.config(width=face_width, height=face_height)

        # Update the UI to apply these changes
        self.root.update_idletasks()

    def _create_panels(self):
        """Create all the panels in the layout."""
        # Create styles for panels
        style = ttk.Style()
        style.configure('Panel.TFrame', background='#000000', borderwidth=1, relief='solid')

        # 1. Camera Feed Panel (Top, full width)
        self.camera_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.camera_panel.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="nsew")

        # Camera label for displaying the feed
        self.camera_label = tk.Label(self.camera_panel, bg='black', width=self.camera_width, height=self.camera_height)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 2. Face Tracking Panels (Middle, 2x2 grid)
        self.face_panel_container = ttk.Frame(self.root, style='Panel.TFrame')
        self.face_panel_container.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="nsew")

        # Configure the face panel container grid
        self.face_panel_container.grid_columnconfigure(0, weight=1)
        self.face_panel_container.grid_columnconfigure(1, weight=1)
        self.face_panel_container.grid_rowconfigure(0, weight=1)
        self.face_panel_container.grid_rowconfigure(1, weight=1)

        # Create 4 face panels in a 2x2 grid
        self.face_panels = []
        self.face_labels = []

        face_width = self.face_panel_width // 2 - 4  # Divide width by 2
        face_height = self.face_panel_height // 2 - 4  # Divide height by 2

        for i in range(4):
            row, col = divmod(i, 2)

            # Create panel frame
            panel = ttk.Frame(self.face_panel_container, style='Panel.TFrame')
            panel.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")

            # Label for face display
            label = tk.Label(panel, bg='black', width=face_width, height=face_height)
            label.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

            self.face_panels.append(panel)
            self.face_labels.append(label)

        # 3. Logging Panel (Bottom Left)
        self.log_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.log_panel.grid(row=2, column=0, padx=2, pady=2, sticky="nsew")

        # Text widget for logs with scrollbar
        self.log_text = tk.Text(self.log_panel, bg='#111111', fg='#CCCCCC',
                               font=('Consolas', 9), wrap=tk.WORD, height=self.log_panel_height//20)
        scrollbar = ttk.Scrollbar(self.log_panel, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)

        # 4. Empty Panel (Bottom Right) for future features
        self.empty_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.empty_panel.grid(row=2, column=1, padx=2, pady=2, sticky="nsew")

        # Placeholder content
        placeholder = ttk.Label(self.empty_panel, text="Reserved for future features",
                              foreground='#888888', background='#222222')
        placeholder.pack(expand=True)

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts for the application."""
        # F11 or Escape to toggle fullscreen
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Ctrl+Q to quit
        self.root.bind("<Control-q>", lambda e: self.root.quit())

        # 'c' to toggle config window
        self.root.bind("<c>", self._on_config_key)

        # 'r' to reset config
        self.root.bind("<r>", self._on_reset_config_key)

        # Bind window resize event
        self.root.bind("<Configure>", self._on_window_resize)

    def _on_config_key(self, event):
        """Handle 'c' key press to toggle config window."""
        if self.config_callback:
            self.config_callback(event)

    def _on_reset_config_key(self, event):
        """Handle 'r' key press to reset config."""
        if self.reset_config_callback:
            self.reset_config_callback(event)

    def _on_window_resize(self, event):
        """Handle window resize events to update panel sizes proportionally."""
        # Only respond to root window resize events, not child widget events
        if event.widget == self.root:
            # Avoid processing during initialization or when minimized
            if event.width > 100 and event.height > 100:
                # Update panel size variables based on new window size
                self.camera_width = event.width - 4  # Full width minus padding
                self.camera_height = int(event.height * 0.5)  # 50% of height
                self.face_panel_width = event.width - 4  # Full width minus padding
                self.face_panel_height = int(event.height * 0.3)  # 30% of height
                self.log_panel_height = int(event.height * 0.15)  # 15% of height

                # Schedule an update of panel sizes (debounced to avoid too many updates)
                if hasattr(self, '_resize_job') and self._resize_job:
                    self.root.after_cancel(self._resize_job)

                # Schedule a new update
                self._resize_job = self.root.after(100, self._update_panel_sizes)

    def _update_panel_sizes(self):
        """Update panel sizes after window resize."""
        # Update row configurations
        self.root.grid_rowconfigure(0, minsize=self.camera_height)
        self.root.grid_rowconfigure(1, minsize=self.face_panel_height)
        self.root.grid_rowconfigure(2, minsize=self.log_panel_height)

        # Update face panel container grid
        self.face_panel_container.grid_columnconfigure(0, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_columnconfigure(1, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_rowconfigure(0, minsize=self.face_panel_height//2)
        self.face_panel_container.grid_rowconfigure(1, minsize=self.face_panel_height//2)

        # Apply the new sizes to panels
        self._set_initial_panel_sizes()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

        # Update panel sizes after toggling fullscreen
        if self.is_fullscreen:
            # In fullscreen, use screen dimensions
            self.camera_width = self.screen_width - 4
            self.camera_height = int(self.screen_height * 0.5)
            self.face_panel_width = self.screen_width - 4
            self.face_panel_height = int(self.screen_height * 0.3)
            self.log_panel_height = int(self.screen_height * 0.15)
        else:
            # In windowed mode, use 80% of screen
            width = int(self.screen_width * 0.8)
            height = int(self.screen_height * 0.8)
            self.camera_width = width - 4
            self.camera_height = int(height * 0.5)
            self.face_panel_width = width - 4
            self.face_panel_height = int(height * 0.3)
            self.log_panel_height = int(height * 0.15)

        # Schedule an update of panel sizes
        self.root.after(100, self._update_panel_sizes)

        return "break"  # Prevent event from propagating

    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes('-fullscreen', False)

            # Update panel sizes for windowed mode
            width = int(self.screen_width * 0.8)
            height = int(self.screen_height * 0.8)
            self.camera_width = width - 4
            self.camera_height = int(height * 0.5)
            self.face_panel_width = width - 4
            self.face_panel_height = int(height * 0.3)
            self.log_panel_height = int(height * 0.15)

            # Schedule an update of panel sizes
            self.root.after(100, self._update_panel_sizes)

        return "break"  # Prevent event from propagating

    def update_camera_feed(self, frame):
        """
        Update the camera feed panel with a new frame.

        Args:
            frame: OpenCV BGR image (numpy array)
        """
        if frame is None:
            return

        try:
            # Convert OpenCV BGR to RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use fixed dimensions for consistent sizing
            panel_width = self.camera_width
            panel_height = self.camera_height

            frame_height, frame_width = rgb_frame.shape[:2]

            # Calculate scaling factor to fit within panel while maximizing size
            scale_width = panel_width / frame_width
            scale_height = panel_height / frame_height
            scale = min(scale_width, scale_height)

            # Calculate new dimensions
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize the frame
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))

            # Create a black background image of panel size
            background = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

            # Calculate position to center the resized frame
            y_offset = (panel_height - new_height) // 2
            x_offset = (panel_width - new_width) // 2

            # Place the resized frame on the black background
            if y_offset >= 0 and x_offset >= 0:
                background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            else:
                # If the frame is larger than the panel, just use the resized frame
                background = resized_frame

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(background)
            self.camera_image = ImageTk.PhotoImage(image=pil_image)

            # Update label with new image
            self.camera_label.configure(image=self.camera_image)
            self.camera_label.image = self.camera_image  # Keep a reference
        except Exception as e:
            print(f"Error updating camera feed: {e}")

    def update_face_panel(self, face_index, face_frame, is_recording=False):
        """
        Update one of the face tracking panels with a new frame.

        Args:
            face_index: Index of the face panel (0-3)
            face_frame: OpenCV BGR image of the face (numpy array)
            is_recording: Whether this face is currently being recorded
        """
        if face_index < 0 or face_index >= 4 or face_frame is None:
            return

        try:
            # Convert OpenCV BGR to RGB for Tkinter
            rgb_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # If recording, add a red circle indicator
            if is_recording:
                # Add red circle to top right corner
                circle_radius = 10
                cv2.circle(rgb_frame,
                          (rgb_frame.shape[1] - circle_radius - 10, circle_radius + 10),
                          circle_radius, (255, 0, 0), -1)

            # Use fixed dimensions for consistent sizing
            face_width = self.face_panel_width // 2 - 4
            face_height = self.face_panel_height // 2 - 4

            # Resize the face frame to fit the panel while maintaining aspect ratio
            frame_height, frame_width = rgb_frame.shape[:2]

            # Calculate scaling factor to fit within panel
            scale_width = face_width / frame_width
            scale_height = face_height / frame_height
            scale = min(scale_width, scale_height)

            # Calculate new dimensions
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize the frame
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))

            # Create a black background image of panel size
            background = np.zeros((face_height, face_width, 3), dtype=np.uint8)

            # Calculate position to center the resized frame
            y_offset = (face_height - new_height) // 2
            x_offset = (face_width - new_width) // 2

            # Place the resized frame on the black background
            if y_offset >= 0 and x_offset >= 0:
                background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            else:
                # If the frame is larger than the panel, just use the resized frame
                background = resized_frame

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(background)
            self.face_images[face_index] = ImageTk.PhotoImage(image=pil_image)

            # Update label with new image
            self.face_labels[face_index].configure(image=self.face_images[face_index])
            self.face_labels[face_index].image = self.face_images[face_index]  # Keep a reference
        except Exception as e:
            print(f"Error updating face panel {face_index}: {e}")

    def clear_face_panel(self, face_index):
        """
        Clear a face panel when a face is no longer detected.

        Args:
            face_index: Index of the face panel (0-3)
        """
        if face_index < 0 or face_index >= 4:
            return

        try:
            # Clear the label
            self.face_labels[face_index].configure(image='')
            self.face_images[face_index] = None
        except Exception as e:
            print(f"Error clearing face panel {face_index}: {e}")

    def add_log_message(self, message):
        """
        Add a message to the logging panel.

        Args:
            message: The message to add
        """
        try:
            # Skip empty messages or messages with only whitespace
            if not message or message.isspace():
                return

            # Enable text widget for editing
            self.log_text.configure(state=tk.NORMAL)

            # Add message with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Strip any trailing newlines to avoid double lines
            message = message.rstrip('\n')

            # Skip empty messages after stripping
            if not message:
                return

            # Format with timestamp and ensure single newline
            formatted_message = f"[{timestamp}] {message}\n"

            # Replace multiple consecutive newlines with a single newline
            formatted_message = re.sub(r'\n+', '\n', formatted_message)

            # Insert at end and scroll to see it
            self.log_text.insert(tk.END, formatted_message)
            self.log_text.see(tk.END)

            # Disable editing
            self.log_text.configure(state=tk.DISABLED)
        except Exception as e:
            print(f"Error adding log message: {e}")

    def clear_log(self):
        """Clear all log messages."""
        try:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.configure(state=tk.DISABLED)
        except Exception as e:
            print(f"Error clearing log: {e}")

    def update(self):
        """
        Process Tkinter events to keep the UI responsive.
        Call this regularly from the main loop.
        """
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception as e:
            print(f"Error updating Tkinter: {e}")

    def run(self):
        """
        Start the Tkinter main loop.
        Only use this if the layout manager owns the root window.
        """
        if self.owns_root:
            # Make sure all widgets are properly sized before entering the main loop
            self.root.update_idletasks()

            # Force geometry to ensure consistent initial size
            if not self.is_fullscreen:
                width = int(self.screen_width * 0.8)
                height = int(self.screen_height * 0.8)
                self.root.geometry(f"{width}x{height}")

            # Apply initial panel sizes
            self._set_initial_panel_sizes()

            # Start the main loop
            self.root.mainloop()

    def quit(self):
        """Close the application."""
        try:
            self.root.quit()
        except Exception as e:
            print(f"Error quitting: {e}")

    def set_config_callback(self, callback):
        """Set the callback function for the config window toggle."""
        self.config_callback = callback

    def set_reset_config_callback(self, callback):
        """Set the callback function for the config reset."""
        self.reset_config_callback = callback


# Simple test function to demonstrate the layout
def test_layout():
    """Create a test layout with placeholder images."""
    import time

    # Create the layout manager
    layout = LayoutManager()

    # Create a test image for the camera feed
    test_camera = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_camera, "Camera Feed", (50, 240),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Create test images for faces
    test_faces = []
    for i in range(4):
        face = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(face, f"Face {i+1}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        test_faces.append(face)

    # Add some log messages
    layout.add_log_message("Application started")
    layout.add_log_message("Initializing camera...")
    layout.add_log_message("Camera initialized successfully")

    # Main loop to update the UI
    try:
        for _ in range(100):  # Run for 100 iterations
            # Update camera feed
            layout.update_camera_feed(test_camera)

            # Update face panels
            for i, face in enumerate(test_faces):
                # Alternate recording status for demonstration
                is_recording = (i % 2 == 0)
                layout.update_face_panel(i, face, is_recording)

            # Add a log message every 10 iterations
            if _ % 10 == 0:
                layout.add_log_message(f"Processing frame {_}")

            # Update the UI
            layout.update()
            time.sleep(0.1)

    except tk.TclError:
        # Handle window being closed
        pass

    print("Test completed")


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_layout()