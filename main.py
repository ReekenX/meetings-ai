# /// script
# dependencies = [
#   "rumps",
# ]
# ///

import rumps
import subprocess
import signal
import threading
import os
import re
import sys
from datetime import datetime

class MeetingRecorderApp(rumps.App):
    def __init__(self):
        super().__init__("Meeting Recorder")
        self.process = None
        self.recording = False
        self.blink_state = False
        self.output_thread = None
        self.meeting_file = None

        # Set up icon paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_active = os.path.join(script_dir, "images", "active.png")
        self.icon_inactive = os.path.join(script_dir, "images", "inactive.png")

        # Set initial icon
        self.icon = self.icon_inactive

        # Create menu items
        self.record_button = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self.menu = [self.record_button]

    def output_reader(self):
        """Read output from recording process and write to both stdout and file."""
        if self.process and self.process.stdout:
            for line in iter(self.process.stdout.readline, b''):
                if not line:
                    break
                line_str = line.decode('utf-8')
                # Print to stdout
                sys.stdout.write(line_str)
                sys.stdout.flush()
                # Write to meeting file
                if self.meeting_file:
                    self.meeting_file.write(line_str)
                    self.meeting_file.flush()

    @rumps.timer(2)
    def blink(self, _):
        """Toggle between active and inactive icon every 2 seconds"""
        if self.recording:
            self.blink_state = not self.blink_state
            self.icon = self.icon_active if self.blink_state else self.icon_inactive

    def toggle_recording(self, sender):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def create_slug(self, title):
        """Create URL-friendly slug from title."""
        # Convert to lowercase
        slug = title.lower()
        # Replace non-alphanumeric characters with hyphens
        slug = re.sub(r'[^a-z0-9-]', '-', slug)
        # Replace multiple hyphens with single hyphen
        slug = re.sub(r'-+', '-', slug)
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        return slug

    def start_recording(self):
        response = rumps.Window(
            message="Enter meeting title:",
            title="Meeting Recorder",
            default_text="",
            ok="Start",
            cancel="Cancel",
            dimensions=(320, 24)
        ).run()

        if response.clicked:
            meeting_title = response.text
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Create slug and filename
            slug = self.create_slug(meeting_title)
            date_str = datetime.now().strftime("%Y-%m-%d")
            meeting_filename = f"{date_str}-{slug}.md"
            meetings_dir = os.path.join(script_dir, "meetings")
            meeting_path = os.path.join(meetings_dir, meeting_filename)

            # Ensure meetings directory exists
            os.makedirs(meetings_dir, exist_ok=True)

            # Create meeting file with title
            with open(meeting_path, 'w') as f:
                f.write(f"# {meeting_title}\n\n")

            # Store current meeting path
            current_meeting_path = os.path.join(script_dir, ".current-meeting")
            with open(current_meeting_path, 'w') as f:
                f.write(meeting_path)

            # Open meeting file for appending
            self.meeting_file = open(meeting_path, 'a')

            # Start recording process - capture output to display and write to file
            record_script = os.path.join(script_dir, "record.py")
            self.process = subprocess.Popen(
                ["uv", "run", record_script, "--model", "tiny", "--duration", "5", "--fp16"],
                stdout=subprocess.PIPE,
                stderr=None  # Let stderr print to console
            )

            # Start thread to read and output the process output
            self.output_thread = threading.Thread(target=self.output_reader, daemon=True)
            self.output_thread.start()

            self.recording = True
            self.record_button.title = "Stop Recording"

    def stop_recording(self):
        self.recording = False

        if self.process:
            try:
                # Try graceful termination first
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If graceful termination fails, force kill
                self.process.kill()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass  # Process is dead or will die soon
            finally:
                self.process = None

        # Wait for output thread to finish reading remaining output
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2)

        # Close meeting file
        if self.meeting_file:
            self.meeting_file.close()
            self.meeting_file = None

        self.icon = self.icon_inactive
        self.record_button.title = "Start Recording"

if __name__ == "__main__":
    MeetingRecorderApp().run()
