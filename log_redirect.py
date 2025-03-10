import sys
import threading
import queue
import datetime
import tkinter as tk

class LogRedirector:
    """
    Redirects stdout and stderr to a Tkinter text widget.
    This allows capturing print statements and other console output
    to display in the application's logging panel.
    """

    def __init__(self, text_widget, max_lines=1000):
        """
        Initialize the log redirector.

        Args:
            text_widget: Tkinter Text widget to display logs
            max_lines: Maximum number of lines to keep in the log (oldest are removed)
        """
        self.text_widget = text_widget
        self.max_lines = max_lines
        self.queue = queue.Queue()
        self.running = True

        # Store original stdout and stderr
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        # Create custom stream objects
        self.stdout_redirector = self.RedirectStream(self, self.stdout)
        self.stderr_redirector = self.RedirectStream(self, self.stderr, is_error=True)

        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def start_redirect(self):
        """Start redirecting stdout and stderr."""
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector

    def stop_redirect(self):
        """Stop redirecting and restore original stdout and stderr."""
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.running = False

    def _update_loop(self):
        """Background thread to update the text widget from the queue."""
        while self.running:
            try:
                # Wait for a message with a timeout to allow checking running flag
                message, is_error = self.queue.get(timeout=0.1)
                self._update_text_widget(message, is_error)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # If there's an error, try to write to the original stderr
                try:
                    self.stderr.write(f"Error in log redirector: {e}\n")
                except:
                    pass

    def _update_text_widget(self, message, is_error=False):
        """Update the text widget with a new message."""
        try:
            # Always schedule the update to run in the main thread
            # This avoids the "main thread is not in main loop" error
            self.text_widget.after(0, lambda: self._direct_update(message, is_error))
        except Exception as e:
            # If there's an error, try to write to the original stderr
            try:
                self.stderr.write(f"Error updating text widget: {e}\n")
            except:
                pass

    def _direct_update(self, message, is_error=False):
        """Directly update the text widget (must be called from main thread)."""
        try:
            # Enable editing
            self.text_widget.config(state=tk.NORMAL)

            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Format the message
            if message.endswith('\n'):
                formatted_message = f"[{timestamp}] {message}"
            else:
                formatted_message = f"[{timestamp}] {message}\n"

            # Set tag for errors (red color)
            tag = "error" if is_error else "normal"

            # Insert the message
            self.text_widget.insert(tk.END, formatted_message, tag)

            # Configure tags if they don't exist
            if "error" not in self.text_widget.tag_names():
                self.text_widget.tag_configure("error", foreground="red")
            if "normal" not in self.text_widget.tag_names():
                self.text_widget.tag_configure("normal", foreground="#CCCCCC")

            # Limit the number of lines
            self._limit_lines()

            # Scroll to the end
            self.text_widget.see(tk.END)

            # Disable editing
            self.text_widget.config(state=tk.DISABLED)
        except Exception as e:
            # If there's an error, try to write to the original stderr
            try:
                self.stderr.write(f"Error in direct update: {e}\n")
            except:
                pass

    def _limit_lines(self):
        """Limit the number of lines in the text widget."""
        line_count = int(self.text_widget.index('end-1c').split('.')[0])
        if line_count > self.max_lines:
            # Delete the oldest lines
            delete_lines = line_count - self.max_lines
            self.text_widget.delete('1.0', f'{delete_lines + 1}.0')

    def write_to_log(self, message, is_error=False):
        """
        Write a message directly to the log.

        Args:
            message: The message to write
            is_error: Whether this is an error message (will be colored red)
        """
        self.queue.put((message, is_error))

    class RedirectStream:
        """Inner class that mimics a file object for redirecting streams."""

        def __init__(self, redirector, original_stream, is_error=False):
            """
            Initialize the redirect stream.

            Args:
                redirector: The parent LogRedirector instance
                original_stream: The original stream (stdout or stderr)
                is_error: Whether this is redirecting stderr
            """
            self.redirector = redirector
            self.original_stream = original_stream
            self.is_error = is_error

        def write(self, string):
            """Write to both the original stream and the queue."""
            # Write to the original stream
            self.original_stream.write(string)

            # Add to the queue for the text widget
            self.redirector.queue.put((string, self.is_error))

        def flush(self):
            """Flush the original stream."""
            self.original_stream.flush()


# Simple test function to demonstrate the log redirector
def test_log_redirect():
    """Test the log redirector with a simple Tkinter window."""
    import time

    # Create a simple Tkinter window
    root = tk.Tk()
    root.title("Log Redirect Test")
    root.geometry("600x400")

    # Create a text widget for logs
    log_text = tk.Text(root, bg='black', fg='white', wrap=tk.WORD)
    log_text.pack(fill=tk.BOTH, expand=True)

    # Create a scrollbar
    scrollbar = tk.Scrollbar(log_text, command=log_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.config(yscrollcommand=scrollbar.set)

    # Create the log redirector
    redirector = LogRedirector(log_text)

    # Start redirecting
    redirector.start_redirect()

    # Print some test messages
    print("This is a normal message")
    print("This is another normal message")

    # Print to stderr
    print("This is an error message", file=sys.stderr)

    # Direct write to log
    redirector.write_to_log("This is a direct message")
    redirector.write_to_log("This is a direct error message", is_error=True)

    # Function to add more messages over time
    def add_messages():
        for i in range(5):
            print(f"Message {i+1} from timer")
            time.sleep(0.5)

        # Stop redirecting when done
        redirector.stop_redirect()
        print("Redirection stopped - this should only appear in console")

    # Start a thread to add messages
    message_thread = threading.Thread(target=add_messages)
    message_thread.daemon = True
    message_thread.start()

    # Start the Tkinter main loop
    root.mainloop()


# Run the test if this file is executed directly
if __name__ == "__main__":
    test_log_redirect()