# Setting Up Camera Access in WSL 2

## Option 1: Using usbipd-win (Recommended for Windows 11)

1. **Install usbipd-win on Windows**:
   - Open PowerShell as Administrator
   - Run: `winget install usbipd`

2. **List available USB devices on Windows**:
   - Open PowerShell as Administrator
   - Run: `usbipd list`
   - Find your webcam in the list and note its BUSID

3. **Attach the webcam to WSL**:
   - In the same PowerShell window, run:
   - `usbipd bind --busid <BUSID>`
   - `usbipd attach --busid <BUSID> --wsl`

4. **Install required packages in WSL**:
   ```bash
   sudo apt update
   sudo apt install linux-tools-generic hwdata
   sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*/usbip 20
   ```

5. **Verify the camera is connected**:
   ```bash
   ls -l /dev/video*
   v4l2-ctl --list-devices
   ```

6. **Run your application**:
   ```bash
   python main.py
   ```

## Option 2: Use a Virtual Camera

If Option 1 doesn't work, you can use a virtual camera solution:

1. Install OBS Studio on Windows
2. Set up a virtual camera in OBS
3. Use the virtual camera in your WSL application

## Option 3: Run the Application in Windows

The simplest solution might be to run your Python application directly in Windows:

1. Install Python on Windows
2. Install the required packages:
   ```
   pip install opencv-python mediapipe
   ```
3. Copy your code to Windows and run it there

## Option 4: Use a Remote Camera

You can also use a smartphone as a remote camera:

1. Install an IP camera app on your smartphone
2. Connect to the camera's IP stream in your code

## Troubleshooting

If you're still having issues, try:

1. Restart WSL: `wsl --shutdown` in PowerShell, then restart your WSL terminal
2. Check if your camera is working in Windows applications
3. Make sure no other application is using the camera
4. Update your WSL to the latest version