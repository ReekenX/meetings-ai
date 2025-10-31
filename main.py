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
from datetime import datetime

class MeetingRecorderApp(rumps.App):
    def __init__(self):
        super().__init__("Meeting Recorder")
        self.process = None
        self.recording = False
        self.blink_state = False
        self.blink_timer = None

        # Set up icon paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.icon_active = os.path.join(script_dir, "icons", "active.png")
        self.icon_inactive = os.path.join(script_dir, "icons", "inactive.png")

        # Set initial icon
        self.icon = self.icon_inactive

        # Create menu items
        self.record_button = rumps.MenuItem("Start Recording", callback=self.toggle_recording)
        self.menu = [self.record_button]

    def blink(self):
        """Toggle between active and inactive icon"""
        if self.recording:
            self.icon = self.icon_active if self.blink_state else self.icon_inactive
            self.blink_state = not self.blink_state
            self.blink_timer = threading.Timer(1.0, self.blink)
            self.blink_timer.start()

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

            # Start recording process - output will be appended to the meeting file
            record_script = os.path.join(script_dir, "record.py")
            with open(meeting_path, 'a') as meeting_file:
                self.process = subprocess.Popen(
                    ["uv", "run", record_script, "--model", "tiny", "--duration", "2", "--fp16"],
                    stdout=meeting_file,
                    stderr=subprocess.PIPE
                )

            self.recording = True
            self.blink()  # Start blinking
            self.record_button.title = "Stop Recording"

    def stop_recording(self):
        self.recording = False
        if self.blink_timer:
            self.blink_timer.cancel()

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

        self.icon = self.icon_inactive
        self.record_button.title = "Start Recording"

if __name__ == "__main__":
    MeetingRecorderApp().run()
