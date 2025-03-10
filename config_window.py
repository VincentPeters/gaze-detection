import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import config
import os

class ConfigWindow:
    def __init__(self):
        """Initialize the Tkinter-based configuration window."""
        # Create the main window but don't show it yet
        self.root = tk.Tk()
        self.root.title("Detection Settings")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.withdraw()  # Hide window initially

        # Flag to track if window is visible
        self.is_visible = False

        # Define settings with clear labels
        self.settings = [
            {"name": "Face Confidence", "config_key": "FACE_DETECTION_CONFIDENCE", "min": 0, "max": 1, "scale": 0.01, "type": "float"},
            {"name": "Face Model", "config_key": "FACE_DETECTION_MODEL", "min": 0, "max": 1, "scale": 1, "type": "int", "options": ["Close Range", "Full Range"]},
            {"name": "Face Margin Percentage", "config_key": "FACE_MARGIN_PERCENT", "min": 0, "max": 100, "scale": 1, "type": "int"},
            {"name": "Redetection Time (seconds)", "config_key": "FACE_REDETECTION_TIMEOUT", "min": 0, "max": 5, "scale": 0.1, "type": "float"},
            {"name": "Eye Contact Threshold", "config_key": "EYE_CONTACT_THRESHOLD", "min": 0, "max": 1, "scale": 0.01, "type": "float"},
            {"name": "Debounce Time (seconds)", "config_key": "DEBOUNCE_TIME", "min": 0, "max": 10, "scale": 0.1, "type": "float"},
            {"name": "Screenshot Debounce (seconds)", "config_key": "SCREENSHOT_DEBOUNCE_TIME", "min": 0, "max": 5, "scale": 0.1, "type": "float"},
            {"name": "Post Gaze Record (seconds)", "config_key": "POST_GAZE_RECORD_TIME", "min": 0, "max": 5, "scale": 0.1, "type": "float"},
            {"name": "High Resolution", "config_key": "HIGH_RES_ENABLED", "min": 0, "max": 1, "scale": 1, "type": "bool"},
            {"name": "Video FPS", "config_key": "VIDEO_FPS", "min": 5, "max": 60, "scale": 1, "type": "int"},
            {"name": "Process Width (pixels)", "config_key": "PROCESSING_WIDTH", "min": 160, "max": 640, "scale": 1, "type": "int"},
            {"name": "Process Height (pixels)", "config_key": "PROCESSING_HEIGHT", "min": 120, "max": 480, "scale": 1, "type": "int"},
            {"name": "Frame Interval (frames)", "config_key": "FRAME_PROCESSING_INTERVAL", "min": 1, "max": 10, "scale": 1, "type": "int"}
        ]

        # Create the UI elements
        self.create_widgets()

        # Dictionary to store slider and value label references
        self.sliders = {}
        self.value_labels = {}

        # Create sliders for each setting
        self.create_sliders()

        # Create preset management section
        self.create_preset_section()

    def create_widgets(self):
        """Create the main UI structure."""
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")

        # Create a canvas with scrollbar for settings
        self.canvas = tk.Canvas(self.settings_frame)
        self.scrollbar = ttk.Scrollbar(self.settings_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Presets tab
        self.presets_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.presets_frame, text="Presets")

        # About tab
        self.about_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.about_frame, text="About")

        # Add about content
        about_text = """
        Eye Contact Detection Settings

        This window allows you to adjust various parameters
        that control the eye contact detection system.

        Keyboard shortcuts:
        - 'c' - Toggle this configuration window
        - 'r' - Reset all settings to defaults
        """
        about_label = ttk.Label(self.about_frame, text=about_text, justify=tk.LEFT)
        about_label.pack(padx=20, pady=20)

        # Add buttons at the bottom
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.reset_button = ttk.Button(self.button_frame, text="Reset All", command=self.reset_config)
        self.reset_button.pack(side=tk.RIGHT, padx=5)

        self.apply_button = ttk.Button(self.button_frame, text="Apply", command=self.apply_settings)
        self.apply_button.pack(side=tk.RIGHT, padx=5)

    def create_sliders(self):
        """Create sliders for each configurable parameter."""
        for i, item in enumerate(self.settings):
            config_key = item["config_key"]
            name = item["name"]
            min_val = item["min"]
            max_val = item["max"]
            scale = item["scale"]
            setting_type = item["type"]

            # Create a frame for this setting
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, padx=10, pady=5)

            # Add label
            label = ttk.Label(frame, text=name, width=25, anchor=tk.W)
            label.pack(side=tk.LEFT)

            # Get current value from config
            current_value = getattr(config, config_key)

            # Create appropriate control based on type
            if setting_type == "bool":
                # Create a checkbox for boolean values
                var = tk.BooleanVar(value=bool(current_value))
                control = ttk.Checkbutton(
                    frame,
                    variable=var,
                    command=lambda v=var, key=config_key: self.update_config_from_checkbox(key, v)
                )
                control.pack(side=tk.LEFT)
                self.sliders[config_key] = var

                # Value display
                value_label = ttk.Label(frame, text=str(bool(current_value)), width=10)
                value_label.pack(side=tk.LEFT, padx=5)
                self.value_labels[config_key] = value_label

            elif setting_type == "int" and "options" in item:
                # Create a combobox for enumerated values
                var = tk.StringVar(value=item["options"][int(current_value)])
                control = ttk.Combobox(
                    frame,
                    textvariable=var,
                    values=item["options"],
                    state="readonly",
                    width=15
                )
                control.pack(side=tk.LEFT)
                control.bind("<<ComboboxSelected>>",
                             lambda e, cb=control, key=config_key, opts=item["options"]:
                             self.update_config_from_combobox(key, cb, opts))
                self.sliders[config_key] = var

                # Value display
                value_label = ttk.Label(frame, text=str(int(current_value)), width=10)
                value_label.pack(side=tk.LEFT, padx=5)
                self.value_labels[config_key] = value_label

            else:
                # Create a slider for numeric values
                var = tk.DoubleVar(value=current_value)
                control = ttk.Scale(
                    frame,
                    from_=min_val,
                    to=max_val,
                    variable=var,
                    command=lambda v, key=config_key, scale=scale, type_=setting_type, var=var:
                            self.update_config_from_slider(key, var.get(), scale, type_)
                )
                control.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.sliders[config_key] = var

                # Value display
                if setting_type == "float":
                    value_text = f"{current_value:.2f}"
                else:
                    value_text = str(int(current_value))

                value_label = ttk.Label(frame, text=value_text, width=10)
                value_label.pack(side=tk.LEFT, padx=5)
                self.value_labels[config_key] = value_label

    def create_preset_section(self):
        """Create the preset management section."""
        # Frame for preset controls
        preset_control_frame = ttk.Frame(self.presets_frame)
        preset_control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Label
        preset_label = ttk.Label(preset_control_frame, text="Configuration Presets:")
        preset_label.pack(side=tk.LEFT, padx=5)

        # Preset name entry
        self.preset_name_var = tk.StringVar()
        preset_entry = ttk.Entry(preset_control_frame, textvariable=self.preset_name_var, width=20)
        preset_entry.pack(side=tk.LEFT, padx=5)

        # Save button
        save_button = ttk.Button(preset_control_frame, text="Save", command=self.save_preset)
        save_button.pack(side=tk.LEFT, padx=5)

        # Frame for preset list
        preset_list_frame = ttk.Frame(self.presets_frame)
        preset_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Listbox with scrollbar for presets
        preset_scrollbar = ttk.Scrollbar(preset_list_frame)
        preset_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.preset_listbox = tk.Listbox(preset_list_frame)
        self.preset_listbox.pack(fill=tk.BOTH, expand=True)

        # Connect scrollbar to listbox
        self.preset_listbox.config(yscrollcommand=preset_scrollbar.set)
        preset_scrollbar.config(command=self.preset_listbox.yview)

        # Load and delete buttons
        preset_button_frame = ttk.Frame(self.presets_frame)
        preset_button_frame.pack(fill=tk.X, padx=10, pady=10)

        load_button = ttk.Button(preset_button_frame, text="Load Selected", command=self.load_preset)
        load_button.pack(side=tk.LEFT, padx=5)

        delete_button = ttk.Button(preset_button_frame, text="Delete Selected", command=self.delete_preset)
        delete_button.pack(side=tk.LEFT, padx=5)

        # Populate preset list
        self.update_preset_list()

    def update_preset_list(self):
        """Update the list of available presets."""
        self.preset_listbox.delete(0, tk.END)
        presets = config.get_config_presets()
        for preset in presets:
            self.preset_listbox.insert(tk.END, preset)

    def save_preset(self):
        """Save current configuration as a preset."""
        preset_name = self.preset_name_var.get().strip()
        if not preset_name:
            messagebox.showerror("Error", "Please enter a preset name")
            return

        try:
            filepath = config.save_config(preset_name)
            messagebox.showinfo("Success", f"Configuration saved as {os.path.basename(filepath)}")
            self.update_preset_list()
            self.preset_name_var.set("")  # Clear the entry
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")

    def load_preset(self):
        """Load selected preset."""
        selection = self.preset_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a preset to load")
            return

        preset_name = self.preset_listbox.get(selection[0])
        try:
            if config.load_config(preset_name):
                messagebox.showinfo("Success", f"Loaded preset: {preset_name}")
                self.update_sliders()
            else:
                messagebox.showerror("Error", f"Failed to load preset: {preset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading preset: {str(e)}")

    def delete_preset(self):
        """Delete selected preset."""
        selection = self.preset_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a preset to delete")
            return

        preset_name = self.preset_listbox.get(selection[0])
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete preset: {preset_name}?"):
            try:
                preset_path = os.path.join(config.CONFIG_PRESETS_DIR, preset_name)
                os.remove(preset_path)
                messagebox.showinfo("Success", f"Deleted preset: {preset_name}")
                self.update_preset_list()
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting preset: {str(e)}")

    def update_config_from_slider(self, config_key, value, scale, type_):
        """Update configuration from slider value."""
        # Convert value based on type
        if type_ == "int":
            value = int(round(value))
        elif type_ == "float":
            value = float(value)

        # Update config
        config.update_config(config_key, value)

        # Update value label
        if type_ == "float":
            self.value_labels[config_key].config(text=f"{value:.2f}")
        else:
            self.value_labels[config_key].config(text=str(value))

    def update_config_from_checkbox(self, config_key, var):
        """Update configuration from checkbox."""
        value = var.get()
        config.update_config(config_key, int(value))
        self.value_labels[config_key].config(text=str(bool(value)))

    def update_config_from_combobox(self, config_key, combobox, options):
        """Update configuration from combobox."""
        selected_text = combobox.get()
        value = options.index(selected_text)
        config.update_config(config_key, value)
        self.value_labels[config_key].config(text=str(value))

    def update_sliders(self):
        """Update all sliders and controls to match current configuration values."""
        for item in self.settings:
            config_key = item["config_key"]
            setting_type = item["type"]

            # Get current value from config
            current_value = getattr(config, config_key)

            # Update control based on type
            if setting_type == "bool":
                self.sliders[config_key].set(bool(current_value))
                self.value_labels[config_key].config(text=str(bool(current_value)))
            elif setting_type == "int" and "options" in item:
                self.sliders[config_key].set(item["options"][int(current_value)])
                self.value_labels[config_key].config(text=str(int(current_value)))
            else:
                self.sliders[config_key].set(current_value)
                if setting_type == "float":
                    self.value_labels[config_key].config(text=f"{current_value:.2f}")
                else:
                    self.value_labels[config_key].config(text=str(int(current_value)))

    def reset_config(self):
        """Reset all configuration parameters to their default values."""
        if messagebox.askyesno("Confirm Reset", "Reset all settings to default values?"):
            config.reset_config()
            self.update_sliders()
            messagebox.showinfo("Reset Complete", "All settings have been reset to default values.")

    def apply_settings(self):
        """Apply current settings."""
        messagebox.showinfo("Settings Applied", "Settings have been applied.")

    def toggle_visibility(self):
        """Toggle the visibility of the configuration window."""
        if self.is_visible:
            self.root.withdraw()
            self.is_visible = False
        else:
            self.root.deiconify()
            self.is_visible = True
            self.update_sliders()  # Refresh values when showing

    def update_window(self):
        """Update the configuration window with current values."""
        # Process Tkinter events
        if self.is_visible:
            self.root.update_idletasks()
            self.root.update()

    def handle_key(self, key):
        """Handle keyboard input for the configuration window."""
        if key == ord('c'):  # 'c' to toggle config window
            self.toggle_visibility()
        elif key == ord('r'):  # 'r' to reset config
            self.reset_config()

    def on_closing(self):
        """Handle window close event."""
        self.toggle_visibility()  # Just hide instead of destroying