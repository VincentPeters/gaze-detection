import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

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
        self.root.configure(background='#333333')

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

        # Define fixed panel sizes based on window size
        self.camera_width = int(width * 0.65) if 'width' in locals() else 640
        self.camera_height = int(height * 0.65) if 'height' in locals() else 480
        self.face_panel_width = int(width * 0.3) if 'width' in locals() else 320
        self.face_panel_height = int(height * 0.3) if 'height' in locals() else 240
        self.log_panel_height = int(height * 0.25) if 'height' in locals() else 200

        # Prevent window from being resized smaller than minimum size
        min_width = self.camera_width + self.face_panel_width + 50  # Add padding
        min_height = self.camera_height + self.log_panel_height + 50  # Add padding
        self.root.minsize(min_width, min_height)

        # Configure grid layout to be responsive
        self.root.grid_columnconfigure(0, weight=2, minsize=self.camera_width)  # Camera feed column (wider)
        self.root.grid_columnconfigure(1, weight=1, minsize=self.face_panel_width)  # Face panels column
        self.root.grid_rowconfigure(0, weight=2, minsize=self.camera_height)     # Top row (taller)
        self.root.grid_rowconfigure(1, weight=1, minsize=self.log_panel_height)     # Bottom row

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
        face_width = self.face_panel_width // 2 - 10  # Account for padding
        face_height = self.camera_height // 2 - 20  # Account for padding and title
        for label in self.face_labels:
            label.config(width=face_width, height=face_height)

        # Update the UI to apply these changes
        self.root.update_idletasks()

    def _create_panels(self):
        """Create all the panels in the layout."""
        # Create styles for panels
        style = ttk.Style()
        style.configure('Panel.TFrame', background='#222222', borderwidth=2, relief='raised')
        style.configure('PanelTitle.TLabel', background='#333333', foreground='white',
                        font=('Arial', 10, 'bold'), padding=5)

        # 1. Camera Feed Panel (Top Left)
        self.camera_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.camera_panel.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Camera panel title
        ttk.Label(self.camera_panel, text="LIVE FEED OF THE CAMERA",
                 style='PanelTitle.TLabel').pack(side=tk.TOP, fill=tk.X)

        # Camera label for displaying the feed (using label instead of canvas)
        self.camera_label = tk.Label(self.camera_panel, bg='black', width=self.camera_width, height=self.camera_height)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 2. Face Tracking Panels (Top Right, 2x2 grid)
        self.face_panel_container = ttk.Frame(self.root, style='Panel.TFrame')
        self.face_panel_container.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Configure the face panel container grid
        self.face_panel_container.grid_columnconfigure(0, weight=1, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_columnconfigure(1, weight=1, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_rowconfigure(0, weight=1, minsize=self.camera_height//2)
        self.face_panel_container.grid_rowconfigure(1, weight=1, minsize=self.camera_height//2)

        # Create 4 face panels in a 2x2 grid
        self.face_panels = []
        self.face_labels = []

        face_titles = ["FACE1 tracking", "FACE2 tracking", "FACE3 tracking", "FACE4 tracking"]
        face_width = self.face_panel_width // 2 - 10  # Account for padding
        face_height = self.camera_height // 2 - 20  # Account for padding and title

        for i in range(4):
            row, col = divmod(i, 2)

            # Create panel frame
            panel = ttk.Frame(self.face_panel_container, style='Panel.TFrame')
            panel.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")

            # Panel title
            ttk.Label(panel, text=face_titles[i], style='PanelTitle.TLabel').pack(side=tk.TOP, fill=tk.X)

            # Label for face display (using label instead of canvas)
            label = tk.Label(panel, bg='black', width=face_width, height=face_height)
            label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

            self.face_panels.append(panel)
            self.face_labels.append(label)

        # 3. Logging Panel (Bottom Left)
        self.log_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.log_panel.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Log panel title
        ttk.Label(self.log_panel, text="LOGGING MESSAGES", style='PanelTitle.TLabel').pack(side=tk.TOP, fill=tk.X)

        # Text widget for logs with scrollbar
        self.log_text = tk.Text(self.log_panel, bg='#111111', fg='#CCCCCC',
                               font=('Consolas', 9), wrap=tk.WORD, height=self.log_panel_height//20)
        scrollbar = ttk.Scrollbar(self.log_panel, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 4. Empty Panel (Bottom Right) for future features
        self.empty_panel = ttk.Frame(self.root, style='Panel.TFrame')
        self.empty_panel.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Empty panel title
        ttk.Label(self.empty_panel, text="EMPTY PANEL", style='PanelTitle.TLabel').pack(side=tk.TOP, fill=tk.X)

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

        # Bind window resize event
        self.root.bind("<Configure>", self._on_window_resize)

    def _on_window_resize(self, event):
        """Handle window resize events to update panel sizes proportionally."""
        # Only respond to root window resize events, not child widget events
        if event.widget == self.root:
            # Avoid processing during initialization or when minimized
            if event.width > 100 and event.height > 100:
                # Update panel size variables based on new window size
                self.camera_width = int(event.width * 0.65)
                self.camera_height = int(event.height * 0.65)
                self.face_panel_width = int(event.width * 0.3)
                self.face_panel_height = int(event.height * 0.3)
                self.log_panel_height = int(event.height * 0.25)

                # Schedule an update of panel sizes (debounced to avoid too many updates)
                # Cancel any existing scheduled update
                if hasattr(self, '_resize_job') and self._resize_job:
                    self.root.after_cancel(self._resize_job)

                # Schedule a new update
                self._resize_job = self.root.after(100, self._update_panel_sizes)

    def _update_panel_sizes(self):
        """Update panel sizes after window resize."""
        # Update column and row configurations
        self.root.grid_columnconfigure(0, minsize=self.camera_width)
        self.root.grid_columnconfigure(1, minsize=self.face_panel_width)
        self.root.grid_rowconfigure(0, minsize=self.camera_height)
        self.root.grid_rowconfigure(1, minsize=self.log_panel_height)

        # Update face panel container grid
        self.face_panel_container.grid_columnconfigure(0, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_columnconfigure(1, minsize=self.face_panel_width//2)
        self.face_panel_container.grid_rowconfigure(0, minsize=self.camera_height//2)
        self.face_panel_container.grid_rowconfigure(1, minsize=self.camera_height//2)

        # Apply the new sizes to panels
        self._set_initial_panel_sizes()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen mode."""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

        # Update panel sizes after toggling fullscreen
        if self.is_fullscreen:
            # In fullscreen, use screen dimensions
            self.camera_width = int(self.screen_width * 0.65)
            self.camera_height = int(self.screen_height * 0.65)
            self.face_panel_width = int(self.screen_width * 0.3)
            self.face_panel_height = int(self.screen_height * 0.3)
            self.log_panel_height = int(self.screen_height * 0.25)
        else:
            # In windowed mode, use 80% of screen
            width = int(self.screen_width * 0.8)
            height = int(self.screen_height * 0.8)
            self.camera_width = int(width * 0.65)
            self.camera_height = int(height * 0.65)
            self.face_panel_width = int(width * 0.3)
            self.face_panel_height = int(height * 0.3)
            self.log_panel_height = int(height * 0.25)

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
            self.camera_width = int(width * 0.65)
            self.camera_height = int(height * 0.65)
            self.face_panel_width = int(width * 0.3)
            self.face_panel_height = int(height * 0.3)
            self.log_panel_height = int(height * 0.25)

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
            panel_height = self.camera_height - 30  # Subtract title height

            frame_height, frame_width = rgb_frame.shape[:2]

            # Calculate scaling factor to fit within panel
            scale_width = panel_width / frame_width
            scale_height = panel_height / frame_height
            scale = min(scale_width, scale_height)

            # Calculate new dimensions
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize the frame
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(resized_frame)
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
            face_width = self.face_panel_width // 2 - 10  # Account for padding
            face_height = self.camera_height // 2 - 20  # Account for padding and title

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

            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(resized_frame)
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
            # Enable text widget for editing
            self.log_text.configure(state=tk.NORMAL)

            # Add message with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"

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